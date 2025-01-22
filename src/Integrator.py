import json
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
from WikidataEmbeddingGenerator import WikidataEmbeddingGenerator
from Matcher import Matcher
from GraphManager import GraphManager
from logger_config import setup_logger
import yaml

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
            matching_config: Configuration for property matching method. If None, uses default config
        """
        self.logger = setup_logger(__name__)
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.properties = {}
        
        # Default matching configuration
        self.matching_config = {
            "use_aliases": True,
            "properties_file": "wikidata-properties-with-aliases.json",
            "entity_query_method": "api",
            "property_query_method": "api",
            "save_matches": False,  # Whether to save matched triples to YAML
            "matches_output_file": "matched_triples.yaml"  # Where to save the matches
        }
        
        # Update with custom config if provided
        if matching_config:
            self.matching_config.update(matching_config)

        # Initialize embedding model and graph manager
        self.embedding_model = SentenceTransformer(embedding_model)
        self.graph_manager = GraphManager()
        self._initialize_configuration()
        
        # Initialize matches storage if saving is enabled
        self.matched_triples = [] if self.matching_config.get("save_matches") else None
    
    def _initialize_configuration(self):
        """Initialize configuration including properties and matcher setup."""
        try:
            properties_file = self.matching_config.get("properties_file")
            use_aliases = self.matching_config.get("use_aliases", True)
            
            generator = WikidataEmbeddingGenerator(embedding_model=self.embedding_model)
            
            if not os.path.exists(properties_file):
                self.logger.info(f"Properties file {properties_file} not found. Generating...")
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
            
            self.logger.info(f"✅ Successfully initialized configuration using {properties_file}")

        except Exception as e:
            self.logger.error(f"❌ Error initializing configuration: {str(e)}")
            raise

    def _process_triple(self, triple: Tuple[str, str, str]) -> None:
        """
        Process a single triple and add it to the graph
        """
        subject, predicate, obj = triple
        
        self.logger.info(f"🔄 Processing triple: (\"{subject}\", \"{predicate}\", \"{obj}\")")
        
        subject_id = self.matcher.query_wikidata_entity(subject)
        if subject_id:
            self.logger.info(f"✓ Found subject ID: {subject_id}")
        else:
            self.logger.warning(f"⚠️ Could not find Wikidata entity for subject: {subject}")
        
        property_id = self.matcher.query_wikidata_property(predicate=predicate)
        if property_id:
            self.logger.info(f"✓ Found property ID: {property_id}")
        else:
            self.logger.warning(f"⚠️ Could not find Wikidata property for predicate: {predicate}")
        
        object_id = self.matcher.query_wikidata_entity(obj)
        if object_id:
            self.logger.info(f"✓ Found object ID: {object_id}")
        else:
            self.logger.warning(f"⚠️ Could not map object to Wikidata: {obj}")
        
        # Store match information if saving is enabled
        if self.matched_triples is not None:
            match_info = {
                'triple': {
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj
                },
                'expected_id': {
                    'subject': subject_id or 'not_found',
                    'predicate': property_id or 'not_found',
                    'object': object_id or 'not_found'
                }
            }
            self.matched_triples.append(match_info)
        
        # Add triple to graph
        self.graph_manager.add_triple(
            subject_id or subject,
            property_id or predicate,
            object_id or obj,
            is_literal=object_id is None
        )

    def _process_triples(self, triples: List[Tuple[str, str, str]]) -> None:
        for triple in triples:
            self._process_triple(triple)

    def _read_triples_from_file(self, file_path: str) -> List[Tuple[str, str, str]]:
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
                            self.logger.warning(f"Warning: Skipping malformed triple in {file_path}: {line}")
                    
                    except Exception as e:
                        self.logger.error(f"Warning: Error parsing line in {file_path}: {line}")
                        self.logger.error(f"Error details: {str(e)}")
                        continue
                        
            return triples
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
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
            self.logger.info(f"Processing file: {file_path.name}")
            
            triples = self._read_triples_from_file(str(file_path))
            self.logger.info(f"Found {len(triples)} triples in {file_path.name}")
            self._process_triples(triples)
            
            file_end_time = time.time()
            file_processing_time = file_end_time - file_start_time
            total_files_time += file_processing_time
            self.logger.info(f"⏱️  File processing time: {file_processing_time:.2f} seconds")
        
        return total_files_time
    
    def process(self) -> None:
        """
        Process all triple files in the input directory structure.
        Handles both flat directory and nested subdirectory structures.
        """
        try:
            self.logger.info("="*50)
            self.logger.info("🔄 Starting Integration Process")
            self.logger.info("="*50)
            
            start_time = time.time()
            initial_triples = len(self.graph_manager.g)
            
            # Check for subdirectories
            subdirs = [d for d in Path(self.input_dir).iterdir() if d.is_dir()]
            
            if subdirs:
                self.logger.info(f"📁 Found {len(subdirs)} subdirectories to process")
                
                for subdir in subdirs:
                    subdir_start_time = time.time()
                    self.logger.info(f"📂 Processing subdirectory: {subdir.name}")
                    txt_files = list(subdir.glob("*_triples.txt"))  # Only look for triples files
                    if not txt_files:
                        self.logger.warning(f"⚠️  No triples files found in {subdir}")
                        continue
                        
                    self.logger.info(f"📄 Found {len(txt_files)} triples files in {subdir.name}")
                    total_files_time = self._process_files(txt_files)
                    
                    subdir_end_time = time.time()
                    subdir_processing_time = subdir_end_time - subdir_start_time
                    
                    # Print subdirectory statistics
                    self.logger.info(f"📊 Subdirectory Statistics for {subdir.name}:")
                    self.logger.info(f"   ⏱️  Total subdirectory time : {subdir_processing_time:.2f} seconds")
                    
                    # Print triple statistics for this subdirectory
                    current_triples = len(self.graph_manager.g)
                    triples_added = current_triples - initial_triples
                    self.logger.info(f"   📈 Triples added in this subdirectory: {triples_added}")
                    if subdir_processing_time > 0:
                        self.logger.info(f"   📈 Processing speed: {triples_added / subdir_processing_time:.2f} triples/second")
                    
                    self.logger.info(f"✅ Completed processing subdirectory: {subdir.name}")
                    initial_triples = current_triples
            else:
                txt_files = list(Path(self.input_dir).glob("*_triples.txt"))
                if not txt_files:
                    self.logger.warning("⚠️  No triples files found in input directory")
                    return
                
                self.logger.info(f"📄 Processing {len(txt_files)} triples files in root directory")
                total_files_time = self._process_files(txt_files)
                self.logger.info(f"📊 Root Directory Statistics:")
                self.logger.info(f"   ⏱️  Total files processing time: {total_files_time:.2f} seconds")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print final statistics
            self._print_statistics("Final Knowledge Graph Statistics")
            
            # Save matches if enabled
            if self.matching_config.get("save_matches"):
                self._save_matches()
            
            self.logger.info(f"⏱️  Total processing time: {total_time:.2f} seconds")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error("❌ Integration failed!")
            raise

    def _save_graph(self, output_file: str, format: str = "turtle") -> None:
        self.graph_manager.save_graph(output_file, format)

    def _load_graph(self, input_file: str, format: str = "turtle") -> None:
        self.graph_manager.load_graph(input_file, format)

    def _get_statistics(self) -> Dict:
        return self.graph_manager.get_statistics()

    def _print_statistics(self, title: str = "Current Statistics") -> None:
        self.graph_manager.print_statistics(title)

    def _save_matches(self) -> None:
        """Save matched triples to YAML file if saving is enabled."""
        if not self.matched_triples:
            return
        
        try:
            output_file = os.path.join(
                self.output_dir, 
                self.matching_config.get("matches_output_file")
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    {'triples': self.matched_triples},
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False
                )
            
            self.logger.info(f"✅ Saved matched triples to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving matched triples: {str(e)}")
            raise


if __name__ == "__main__":
    example_triples = [
    ("Artificial intelligence", "is a field of research in", "computer science"),  # P19: place of birth
    ("Albert Einstein", "died in", "Princeton"),  # P20: place of death
    ("Albert Einstein", "worked on", "Theory of Relativity"),  # P800: notable work
    ("Albert Einstein", "received", "Nobel Prize in Physics"),  # P166: award received
    ]
    
    pipeline = Integrator(input_dir="./output_model", output_dir="./output_integrator", embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    pipeline.process()
    