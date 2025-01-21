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
import torch
from .WikidataEmbeddingGenerator import WikidataEmbeddingGenerator
from .Matcher import Matcher

class Integrator:
    """
    Class for integrating triples into the Wikidata Ontology.
    """
    def __init__(self, input_dir: str, output_dir: str, embedding_model: str, matching_config: Dict = None):
        """
        Initialize the Integrator with input and output directories.
        
        Args:
            input_dir: Directory containing input triple files (.txt)
            output_dir: Directory where the final knowledge graph will be saved
            embedding_model: Name of the sentence transformer model to use
            matching_config: Configuration for property matching method
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialize caches and properties
        self.entity_cache = {}
        self.property_cache = {}
        self.properties = {}

        # Set up matching configuration
        self.matching_config = matching_config or {
            "use_aliases": True,
            "properties_file": "wikidata-properties-with-aliases.json"
        }

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self._initialize_knowledge_graph()
        self._initialize_properties()
    
    def _initialize_knowledge_graph(self):
        """Initialize the RDF graph and set up Wikidata namespaces."""
        try:
            self.g = Graph()

            self.wd = Namespace("http://www.wikidata.org/entity/")
            self.wdt = Namespace("http://www.wikidata.org/prop/direct/")
            
            self.g.bind("wd", self.wd)
            self.g.bind("wdt", self.wdt)
            
            print("✅ Successfully initialized knowledge graph and namespaces")

        except Exception as e:
            print(f"❌ Error initializing knowledge graph: {str(e)}")
            raise

    def _initialize_properties(self):
        """Initialize properties based on matching configuration."""
        try:
            properties_file = self.matching_config.get("properties_file")
            use_aliases = self.matching_config.get("use_aliases", True)
            
            generator = WikidataEmbeddingGenerator(embedding_model=self.embedding_model)
            
            if not os.path.exists(properties_file):
                print(f"Properties file {properties_file} not found. Generating...")
                self.properties = generator.generate_embeddings(
                    output_file=properties_file,
                    use_aliases=use_aliases
                )
            else:
                self.properties = generator.load_embeddings(
                    file_path=properties_file,
                    use_aliases=use_aliases
                )

            self.matcher = Matcher(
                embedding_model=self.embedding_model,
                properties=self.properties,
                matching_config=self.matching_config
            )
            
            print(f"✅ Successfully initialized properties using {properties_file}")

        except Exception as e:
            print(f"❌ Error initializing properties: {str(e)}")
            raise

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
                
            return None
            
        except Exception as e:

            print(f"Error searching for entity {label}: {e}")

    def query_wikidata_property(self, predicate: str, method: str = "sparql", file_path: str = "wikidata-properties-with-aliases.json", language: str = "en") -> str:
        """
        Search for Wikidata properties using either SPARQL or the Wikidata API.
        
        Args:
            predicate: The predicate text to search for
            method: "sparql" or "api" for the search method
            file_path: Path to property cache file (only for SPARQL)
            language: Language code (default: "en")
        
        Returns:
            Property ID or None if not found
        """
        if method.lower() == "sparql":
            best_match = self.matcher.find_best_match(predicate)
            return best_match
        
        elif method.lower() == "api":
            # API implementation remains unchanged
            if predicate in self.property_cache:
                return self.property_cache[predicate]

            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "format": "json",
                "language": language,
                "search": predicate,
                "type": "property"
            }
            
            try:
                response = requests.get(url, params=params)
                sleep(0.1)  # Rate limiting
                data = response.json()
                
                if data["search"]:
                    # Check each result for exact matches
                    for result in data["search"]:
                        # Check main label
                        if result.get("label", "").lower() == predicate.lower():
                            property_id = result["id"]
                            self.property_cache[predicate] = property_id
                            print(f"Found exact match for '{predicate}': {property_id} ({result.get('label')})")
                            return property_id
                    
                    # If no exact match, take the first result
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

    def process_triple(self, triple: Tuple[str, str, str]) -> None:
        """
        Process a single triple and add it to the graph
        """
        subject, predicate, obj = triple
        
        print(f"\n🔄 Processing triple: (\"{subject}\", \"{predicate}\", \"{obj}\")")
        
        subject_id = self.query_wikidata_entity(subject)
        if subject_id:
            print(f"✓ Found subject ID: {subject_id}")
        else:
            print(f"⚠️  Could not find Wikidata entity for subject: {subject}")
        
        property_id = self.query_wikidata_property(predicate=predicate)
        if property_id:
            print(f"✓ Found property ID: {property_id}")
        else:
            print(f"⚠️  Could not find Wikidata property for predicate: {predicate}")
        
        object_id = self.query_wikidata_entity(obj)
        if object_id:
            print(f"✓ Found object ID: {object_id}")
        else:
            print(f"⚠️  Could not map object to Wikidata: {obj}")
        
        subject_uri = self.wd[subject_id] if subject_id else URIRef(f"http://example.org/entity/{quote(subject)}")
        encoded_predicate = quote(predicate.lower().replace(" ", "_"))
        property_uri = self.wdt[property_id] if property_id else URIRef(f"http://example.org/property/{encoded_predicate}")
        
        if object_id:
            object_uri = self.wd[object_id]
            self.g.add((subject_uri, property_uri, object_uri))
        else:
            self.g.add((subject_uri, property_uri, Literal(obj)))

    def print_statistics(self, title: str = "Current Statistics") -> None:
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print(f"📊 {title}")
        print("="*50)
 
        max_num_width = max(len(str(v)) for v in stats.values())
        
        print(f"\n🔸 Total Triples:       {stats['total_triples']:>{max_num_width}}")
        print(f"🔸 Unique Subjects:     {stats['unique_subjects']:>{max_num_width}}")
        print(f"🔸 Unique Predicates:   {stats['unique_predicates']:>{max_num_width}}")
        print(f"🔸 Unique Objects:      {stats['unique_objects']:>{max_num_width}}\n")
        
        
        print("\n" + "="*50)

    def process_directory(self) -> None:
        try:
            print("\n" + "="*50)
            print("🔄 Starting Integration Process")
            print("="*50)
            
            start_time = time.time()
            initial_triples = len(self.g)
            
            # Check for subdirectories
            subdirs = [d for d in Path(self.input_dir).iterdir() if d.is_dir()]
            
            if subdirs:
                print(f"📁 Found {len(subdirs)} subdirectories to process")
                
                for subdir in subdirs:
                    subdir_start_time = time.time()
                    print(f"\n📂 Processing subdirectory: {subdir.name}")
                    txt_files = list(subdir.glob("*_triples.txt"))  # Only look for triples files
                    if not txt_files:
                        print(f"⚠️  No triples files found in {subdir}")
                        continue
                        
                    print(f"📄 Found {len(txt_files)} triples files in {subdir.name}")
                    total_files_time = self._process_files(txt_files)
                    
                    subdir_end_time = time.time()
                    subdir_processing_time = subdir_end_time - subdir_start_time
                    
                    # Print subdirectory statistics
                    print(f"\n📊 Subdirectory Statistics for {subdir.name}:")
                    print(f"   ⏱️  Total subdirectory time : {subdir_processing_time:.2f} seconds")
                    
                    # Print triple statistics for this subdirectory
                    current_triples = len(self.g)
                    triples_added = current_triples - initial_triples
                    print(f"   📈 Triples added in this subdirectory: {triples_added}")
                    if subdir_processing_time > 0:
                        print(f"   📈 Processing speed: {triples_added / subdir_processing_time:.2f} triples/second")
                    
                    print(f"✅ Completed processing subdirectory: {subdir.name}")
                    initial_triples = current_triples
            else:
                txt_files = list(Path(self.input_dir).glob("*_triples.txt"))
                if not txt_files:
                    print("⚠️  No triples files found in input directory")
                    return
                
                print(f"📄 Processing {len(txt_files)} triples files in root directory")
                total_files_time = self._process_files(txt_files)
                print(f"\n📊 Root Directory Statistics:")
                print(f"   ⏱️  Total files processing time: {total_files_time:.2f} seconds")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print final statistics
            self.print_statistics("Final Knowledge Graph Statistics\n")
            
            print(f"\n⏱️  Total processing time: {total_time:.2f} seconds\n")
            print("="*50 + "\n")
            
        except Exception as e:
            print("❌ Integration failed!")
            raise

    def process_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        for triple in triples:
            self.process_triple(triple)

    def save_graph(self, output_file: str, format: str = "turtle") -> None:
        self.g.serialize(destination=output_file, format=format)

    def load_graph(self, input_file: str, format: str = "turtle") -> None:
        self.g.parse(input_file, format=format)

    def get_statistics(self) -> Dict:
        return {
            "total_triples": len(self.g),
            "unique_subjects": len(set(self.g.subjects())),
            "unique_predicates": len(set(self.g.predicates())),
            "unique_objects": len(set(self.g.objects()))
        }

    def read_triples_from_file(self, file_path: str) -> List[Tuple[str, str, str]]:
        """
        Read triples from a text file.
        Expected format: Each line should contain a triple in the format ("Subject Name", "Property Name", "Object name");
        
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
                    
                    # Remove trailing semicolon if present
                    if line.endswith(';'):
                        line = line[:-1]
                    
                    # Remove outer parentheses
                    line = line.strip('()')
                    
                    try:
                        # Split by comma but preserve commas within quotes
                        parts = []
                        current = []
                        in_quotes = False
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                parts.append(''.join(current).strip())
                                current = []
                                continue
                            current.append(char)
                        
                        # Add the last part
                        if current:
                            parts.append(''.join(current).strip())
                        
                        # Clean up each part (remove quotes and extra whitespace)
                        parts = [p.strip().strip('"') for p in parts]
                        
                        if len(parts) == 3:
                            subject, predicate, obj = parts
                            triples.append((subject, predicate, obj))
                        else:
                            print(f"Warning: Skipping malformed triple in {file_path}: {line}")
                    
                    except Exception as e:
                        print(f"Warning: Error parsing line in {file_path}: {line}")
                        print(f"Error details: {str(e)}")
                        continue
                        
            return triples
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return []

    def _process_files(self, files) -> None:
        """
        Process a list of files.
        
        Args:
            files: List of Path objects pointing to files to process
        Returns:
            float: Total processing time for all files
        """
        total_files_time = 0
        
        # Filter for only triples files and sort by chunk number
        triples_files = [f for f in files if f.name.endswith('_triples.txt')]
        triples_files.sort(key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else float('inf'))
        
        for file_path in triples_files:
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