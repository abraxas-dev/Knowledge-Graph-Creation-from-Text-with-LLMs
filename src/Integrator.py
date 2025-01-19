import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from typing import List, Tuple, Dict, Union
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from time import sleep
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os 
from pathlib import Path
import time

class Integrator:
    def __init__(self, input_dir: str, output_dir: str, embedding_model: str):
        """
        Initialize the Integrator with input and output directories.
        
        Args:
            input_dir: Directory containing input triple files (.txt)
            output_dir: Directory where the final knowledge graph will be saved
            embedding_model: Name of the sentence transformer model to use
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.g = Graph()
        self.wd = Namespace("http://www.wikidata.org/entity/")
        self.wdt = Namespace("http://www.wikidata.org/prop/direct/")
        self.g.bind("wd", self.wd)
        self.g.bind("wdt", self.wdt)
        
        self.entity_cache = {}
        self.property_cache = {}

        self.embedding_model = SentenceTransformer(embedding_model)
        self.properties = {}
    
    def load_wikidata_properties(self, output_file="wikidata-properties.json") -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """
        Lädt alle Wikidata-Properties mit Labels, Beschreibungen, Aliassen und generiert Embeddings
        !!! Es funktioniert zurzeit nicht ganz !

        For aliases :
        SELECT ?property ?propertyLabel ?propertyDescription ?altLabel
        WHERE {
        ?property a wikibase:Property .
        SERVICE wikibase:label { 
        bd:serviceParam wikibase:language "en" .
        ?property rdfs:label ?propertyLabel .
        }
        OPTIONAL { ?property schema:description ?propertyDescription . FILTER(LANG(?propertyDescription) = "en") }
        OPTIONAL { ?property skos:altLabel ?altLabel . FILTER(LANG(?altLabel) = "en") }
        }
        """

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.addCustomHttpHeader('User-Agent', 'Bot/1.0')
        query = """
        SELECT ?property ?propertyLabel
        WHERE {
          ?property a wikibase:Property .
          SERVICE wikibase:label { 
            bd:serviceParam wikibase:language "en" .
            ?property rdfs:label ?propertyLabel .
          }
        }
        """
        
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(60)
        properties = {}

        try:
            results = sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                prop_id = result["property"]["value"].split("/")[-1]
                label = result["propertyLabel"]["value"]

                embedding = self.embedding_model.encode(label).tolist()
                properties[prop_id] = {
                    "label": label,
                    "embedding": embedding
                }
            
            with open(output_file, "w") as file:
                json.dump(properties, file, indent=4)
            print(f"Properties saved to {output_file}")
        
        except Exception as e:
            print(f"Error fetching properties: {e}")

    def load_embeddings(self, file_path="wikidata-properties.json"):
        """
        Lädt die gespeicherten Embeddings aus einer Datei und speichert sie in einem Dictionary.
        """
        try:
            with open(file_path, "r") as file:
                self.properties = json.load(file)
            
            print(f"Successfully loaded {len(self.properties)} properties from {file_path}")
            # For testing :
            #for prop_id, prop_data in self.properties.items():
            #    label = prop_data["label"]  # Hole das Label
            #    print(f"ID: {prop_id}, Label: {label}")
            return self.properties
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return {}
        
    def find_best_match(self, predicate: str, file_path="wikidata-properties.json"):
        """
        Berechnet den Best-Match für ein Prädikat basierend auf der Kosinus-Ähnlichkeit.
        """
        
        if not self.properties:
            print("No properties loaded. Please ensure the file exists or run 'load_wikidata_properties' first.")
            return None
        
        # Berechne das Embedding für das gesuchte Prädikat
        predicate_embedding = self.embedding_model.encode(predicate)

        best_match = None
        highest_similarity = -1

        # Vergleiche mit allen gespeicherten Properties
        for prop_id, prop_data in self.properties.items():
            stored_embedding = prop_data["embedding"]
            similarity = util.cos_sim(predicate_embedding, stored_embedding).item()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {
                    "property_id": prop_id,
                    "label": prop_data["label"],
                    "similarity": highest_similarity
                }
        
        return best_match

    def query_wikidata_entity(self, label: str, language: str = "en") -> str:
        """
        Searches for a Wikidata entity based on a label, including alternative labels (also known as).
        
        Args:
            label: The label to search for
            language: The language code (default: "en")
            
        Returns:
            The Wikidata entity ID if found, None otherwise
        """
        if label in self.entity_cache:
            return self.entity_cache[label]

        # First, try exact search with the main API
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "search": label,
            "type": "item"
        }
        
        try:
            response = requests.get(url, params=params)
            sleep(0.1)  # Rate limiting
            data = response.json()
            
            if data["search"]:
                # Check each result for exact matches in labels or aliases
                for result in data["search"]:
                    # Check main label
                    if result.get("label", "").lower() == label.lower():
                        entity_id = result["id"]
                        self.entity_cache[label] = entity_id
                        return entity_id
                    
                    # Check aliases if present
                    aliases = result.get("aliases", [])
                    for alias in aliases:
                        if alias.lower() == label.lower():
                            entity_id = result["id"]
                            self.entity_cache[label] = entity_id
                            return entity_id
                
                # If no exact match found, return the first result
                entity_id = data["search"][0]["id"]
                self.entity_cache[label] = entity_id
                return entity_id
                
            return -1
            
        except Exception as e:
            print(f"Error searching for entity {label}: {e}")

    def query_wikidata_property(self, predicate: str, method: str = "sparql", file_path: str = "wikidata-properties.json", language: str = "en") -> str:
        """
        Sucht nach Wikidata-Properties entweder über SPARQL oder die Wikidata-API.
        
        Args:
            predicate: Der zu suchende Prädikat-Text
            method: "sparql" oder "api" für die Suchmethode
            file_path: Pfad zur Property-Cache-Datei (nur für SPARQL)
            language: Sprachcode (default: "en")
        
        Returns:
            Property-ID oder None wenn nicht gefunden
        """
        if method.lower() == "sparql":
            # Bestehende SPARQL-Implementierung
            if not os.path.exists(file_path):
                print(f"File '{file_path}' not found. Generating properties...")
                self.load_wikidata_properties(output_file=file_path)

            self.load_embeddings(file_path=file_path)
            best_match = self.find_best_match(predicate)

            if best_match:
                print(f"Best Match for '{predicate}':")
                print(f"Property ID: {best_match['property_id']}")
                print(f"Label: {best_match['label']}")
                print(f"Similarity: {best_match['similarity']:.4f}")
                return best_match["property_id"]
            
        elif method.lower() == "api":
            # Neue API-Implementierung
            if predicate in self.property_cache:
                return self.property_cache[predicate]

            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "format": "json",
                "language": language,
                "search": predicate,
                "type": "property"  # Suche nach Properties statt Items
            }
            
            try:
                response = requests.get(url, params=params)
                sleep(0.1)  # Rate limiting
                data = response.json()
                
                if data["search"]:
                    # Überprüfe jedes Ergebnis auf exakte Übereinstimmungen
                    for result in data["search"]:
                        # Prüfe Hauptlabel
                        if result.get("label", "").lower() == predicate.lower():
                            property_id = result["id"]
                            self.property_cache[predicate] = property_id
                            print(f"Found exact match for '{predicate}': {property_id} ({result.get('label')})")
                            return property_id
                    
                    # Wenn keine exakte Übereinstimmung, nimm das erste Ergebnis
                    property_id = data["search"][0]["id"]
                    label = data["search"][0].get("label", "")
                    self.property_cache[predicate] = property_id
                    print(f"Found closest match for '{predicate}': {property_id} ({label})")
                    return property_id
                
                print(f"No property found for '{predicate}'")
                return None
                
            except Exception as e:
                print(f"Error searching for property '{predicate}': {e}")
                return None
        
        else:
            raise ValueError(f"Invalid method: {method}. Use 'sparql' or 'api'.")

        print(f"No match found for predicate '{predicate}'.")
        return None

    def process_triple(self, triple: Tuple[str, str, str]) -> None:
        """
        Verarbeitet ein einzelnes Tripel und fügt es dem Graphen hinzu
        """
        subject, predicate, obj = triple
        
        print(f"\nProcessing triple: {subject} - {predicate} - {obj}")
        
        subject_id = self.query_wikidata_entity(subject)
        print(f"Found subject ID: {subject_id}")
        
        property_id = self.query_wikidata_property(predicate=predicate)
        print(f"Found property ID: {property_id}")
        
        object_id = self.query_wikidata_entity(obj)
        print(f"Found object ID: {object_id}")
        
        subject_uri = self.wd[subject_id] if subject_id else URIRef(f"http://example.org/entity/{quote(subject)}")
        encoded_predicate = quote(predicate.lower().replace(" ", "_"))
        property_uri = self.wdt[property_id] if property_id else URIRef(f"http://example.org/property/{encoded_predicate}")
        
        if object_id:
            object_uri = self.wd[object_id]
            self.g.add((subject_uri, property_uri, object_uri))
            print(f"Added triple with entity object: {subject_uri} - {property_uri} - {object_uri}")
        else:
            self.g.add((subject_uri, property_uri, Literal(obj)))
            print(f"Added triple with literal object: {subject_uri} - {property_uri} - {obj}")

        if not subject_id:
            print(f"Warning: Could not find Wikidata entity for subject: {subject}")
        if not property_id:
            print(f"Warning: Could not find Wikidata property for predicate: {predicate}")

    def process_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        """
        Verarbeitet eine Liste von Tripeln
        """
        for triple in triples:
            self.process_triple(triple)

    def save_graph(self, output_file: str, format: str = "turtle") -> None:
        """
        Speichert den RDF-Graphen in einer Datei
        """
        self.g.serialize(destination=output_file, format=format)

    def load_graph(self, input_file: str, format: str = "turtle") -> None:
        """
        Lädt einen existierenden RDF-Graphen aus einer Datei
        """
        self.g.parse(input_file, format=format)

    def get_statistics(self) -> Dict:
        """
        Erstellt Statistiken über den Graphen
        """
        return {
            "total_triples": len(self.g),
            "unique_subjects": len(set(self.g.subjects())),
            "unique_predicates": len(set(self.g.predicates())),
            "unique_objects": len(set(self.g.objects()))
        }

    def read_triples_from_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Read triples from a text file.
        Expected format: Each line should contain a triple in the format "<subject> <predicate> <object>"
        
        Args:
            file_path: Path to the text file containing triples
            
        Returns:
            List of tuples, where each tuple contains (subject, predicate, object)
        """
        triples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    
                    # Remove angle brackets if present
                    line = line.replace('<', '').replace('>', '')
                    
                    # Split the line into subject, predicate, object
                    parts = line.split(' ', 2)
                    if len(parts) == 3:
                        subject, predicate, obj = parts
                        triples.append((subject.strip(), predicate.strip(), obj.strip()))
                    else:
                        print(f"Warning: Skipping malformed triple in {file_path}: {line}")
                        
            return triples
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return []

    def process_directory(self) -> None:
        try:
            print("\n" + "="*50)
            print("🔄 Starting Integration Process")
            print("="*50)
            
            start_time = time.time()
            
            # Check for subdirectories
            subdirs = [d for d in Path(self.input_dir).iterdir() if d.is_dir()]
            
            if subdirs:
                print(f"📁 Found {len(subdirs)} subdirectories to process")
                
                for subdir in subdirs:
                    subdir_start_time = time.time()
                    print(f"\n📂 Processing subdirectory: {subdir.name}")
                    txt_files = list(subdir.glob("*.txt"))
                    if not txt_files:
                        print(f"⚠️  No .txt files found in {subdir}")
                        continue
                        
                    print(f"📄 Found {len(txt_files)} files in {subdir.name}")
                    total_files_time = self._process_files(txt_files)
                    
                    subdir_end_time = time.time()
                    subdir_processing_time = subdir_end_time - subdir_start_time
                    print(f"\n📊 Subdirectory Statistics for {subdir.name}:")
                    print(f"   ⏱️  Total files processing time: {total_files_time:.2f} seconds")
                    print(f"   ⏱️  Total subdirectory time (including overhead): {subdir_processing_time:.2f} seconds")
                    print(f"✅ Completed processing subdirectory: {subdir.name}")
            else:
                txt_files = [f for f in Path(self.input_dir).glob("*.txt")]
                if not txt_files:
                    print("⚠️  No .txt files found in input directory")
                    return
                
                print(f"📄 Processing {len(txt_files)} files in root directory")
                total_files_time = self._process_files(txt_files)
                print(f"\n📊 Root Directory Statistics:")
                print(f"   ⏱️  Total files processing time: {total_files_time:.2f} seconds")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n" + "="*50)
            print(f"✅ Integration Complete!")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print("="*50 + "\n")
            
        except Exception as e:
            print("❌ Integration failed!")
            raise

    def _process_files(self, files) -> None:
        """
        Process a list of files.
        
        Args:
            files: List of Path objects pointing to files to process
        """
        total_files_time = 0
        
        for file_path in files:
            file_start_time = time.time()
            print(f"\nProcessing file: {file_path.name}")
            
            triples = self.read_triples_from_file(str(file_path))
            print(f"Found {len(triples)} triples in {file_path.name}")
            self.process_triples(triples)
            
            file_end_time = time.time()
            file_processing_time = file_end_time - file_start_time
            total_files_time += file_processing_time
            print(f"⏱️  File processing time: {file_processing_time:.2f} seconds")
        
        return total_files_time

if __name__ == "__main__":
    example_triples = [
    ("Artificial intelligence", "is a field of research in", "computer science"),  # P19: place of birth
    ("Albert Einstein", "died in", "Princeton"),  # P20: place of death
    ("Albert Einstein", "worked on", "Theory of Relativity"),  # P800: notable work
    ("Albert Einstein", "received", "Nobel Prize in Physics"),  # P166: award received
    
    ]
    
    pipeline = Integrator(input_dir="../data/triples", output_dir="../data/knowledge_graph", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    #pipeline.load_wikidata_properties()
    pipeline.process_triples(example_triples)

    #pipeline.process_directory()
    
    stats = pipeline.get_statistics()
    print("Statistiken:", json.dumps(stats, indent=2))
    
    pipeline.save_graph("../data/knowledge_graph/stats")