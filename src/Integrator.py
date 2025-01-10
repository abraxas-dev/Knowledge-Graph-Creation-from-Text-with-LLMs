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

class Integrator:
    def __init__(self):
        self.g = Graph()
        self.wd = Namespace("http://www.wikidata.org/entity/")
        self.wdt = Namespace("http://www.wikidata.org/prop/direct/")
        self.g.bind("wd", self.wd)
        self.g.bind("wdt", self.wdt)
        
        self.entity_cache = {}
        self.property_cache = {}

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
        Sucht nach einer Wikidata-Entität basierend auf einem Label
        """
        if label in self.entity_cache:
            return self.entity_cache[label]

        url = f"https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "search": label,
            "type": "item"
        }
        
        try:
            response = requests.get(url, params=params)
            sleep(0.1) 
            data = response.json()
            
            if data["search"]:
                entity_id = data["search"][0]["id"]
                self.entity_cache[label] = entity_id
                return entity_id
            return None
        except Exception as e:
            print(f"Fehler beim Suchen der Entität {label}: {e}")
            return None

    def query_wikidata_property(self, predicate: str, file_path: str = "wikidata-properties.json",language: str = "en") -> str:
        """
        Überprüft, ob die Datei mit den Properties existiert. Wenn ja, lädt sie die Properties
        und führt die Best-Match-Methode aus. Falls nicht, ruft sie 'load_wikidata_properties' auf.
        """
        if not os.path.exists(file_path):
            print(f"File '{file_path}' not found. Generating properties...")
            self.load_wikidata_properties(output_file=file_path)

        # Lade die Properties und führe die Best-Match-Methode aus
        self.load_embeddings(file_path=file_path)
        best_match = self.find_best_match(predicate)

        if best_match:
            print(f"Best Match for '{predicate}':")
            print(f"Property ID: {best_match['property_id']}")
            print(f"Label: {best_match['label']}")
            print(f"Similarity: {best_match['similarity']:.4f}")
            return best_match["property_id"]
        else:
            print(f"No match found for predicate '{predicate}'.")

    def process_triple(self, triple: Tuple[str, str, str]) -> None:
        """
        Verarbeitet ein einzelnes Tripel und fügt es dem Graphen hinzu
        """
        subject, predicate, obj = triple
        
        print(f"\nProcessing triple: {subject} - {predicate} - {obj}")
        
        subject_id = self.query_wikidata_entity(subject)
        print(f"Found subject ID: {subject_id}")
        
        property_id = self.query_wikidata_property(predicate)
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

if __name__ == "__main__":
    example_triples = [
    ("Albert Einstein", "born in", "Ulm"),  # P19: place of birth
    ("Albert Einstein", "died in", "Princeton"),  # P20: place of death
    ("Albert Einstein", "worked on", "Theory of Relativity"),  # P800: notable work
    ("Albert Einstein", "received", "Nobel Prize in Physics"),  # P166: award received
    
    ]
    
    pipeline = Integrator()
    pipeline.load_wikidata_properties()
    #pipeline.load_embeddings()
    pipeline.process_triples(example_triples)
    
    stats = pipeline.get_statistics()
    print("Statistiken:", json.dumps(stats, indent=2))
    
    pipeline.save_graph("wikidata_knowledge_graph.ttl")