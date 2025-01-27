"""
Project : Knowledge Graph Creation from Text
Author : @abraxas-dev
"""
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer
import json
import os
from typing import Dict, Union, List
from time import sleep
from src.utils.logger_config import setup_logger

class WikidataEmbeddingGenerator:
    """
    Class for generating and managing Wikidata property/entity(?) embeddings, including their aliases.
    """
    def __init__(self, embedding_model):
        """
        Initialize the WikidataEmbeddingGenerator.
        
        Args:
            embedding_model: The initialised sentence transformer model from Integrator
        """
        self.logger = setup_logger(__name__)
        self.embedding_model = embedding_model
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.addCustomHttpHeader('User-Agent', 'Bot/4.2')
        self.sparql.setReturnFormat(JSON)
        self.sparql.setTimeout(60)

    def generate_embeddings(self, output_file: str = "wikidata-properties.json", use_aliases: bool = True) -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """
        Generate embeddings for Wikidata properties with or without aliases.
        
        Args:
            output_file: Path to save the properties with their embeddings
            use_aliases: Whether to include aliases in the property data
            
        Returns:
            Dictionary containing property information with embeddings
        """
        if use_aliases:
            return self._generate_embeddings_with_aliases(output_file)
        else:
            return self._generate_embeddings_simple(output_file)

    def _generate_embeddings_simple(self, output_file: str) -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """Generate embeddings for properties without aliases."""
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
        
        properties = {}
        try:
            self.logger.info("🔄 Fetching Wikidata properties...")
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            self.logger.info("✨ Computing embeddings...")
            for result in results["results"]["bindings"]:
                prop_id = result["property"]["value"].split("/")[-1]
                label = result["propertyLabel"]["value"]
                
                embedding = self.embedding_model.encode(label).tolist()
                properties[prop_id] = {
                    "label": label,
                    "embedding": embedding
                }
            
            self.logger.info(f"📝 Saving properties to {output_file}...")
            with open(output_file, "w", encoding='utf-8') as file:
                json.dump(properties, file, indent=4, ensure_ascii=False)
            
            self.logger.info(f"✅ Successfully saved {len(properties)} properties")
            return properties
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching properties: {str(e)}")
            raise

    def _generate_embeddings_with_aliases(self, output_file: str) -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """Generate embeddings for properties with aliases."""
        query = """
        SELECT ?property ?propertyLabel ?altLabel
        WHERE {
          ?property a wikibase:Property .
          SERVICE wikibase:label { 
            bd:serviceParam wikibase:language "en" .
            ?property rdfs:label ?propertyLabel .
          }
          OPTIONAL { 
            ?property skos:altLabel ?altLabel . 
            FILTER(LANG(?altLabel) = "en")
          }
        }
        """
        
        properties = {}
        try:
            self.logger.info("🔄 Fetching Wikidata properties with aliases...")
            self.sparql.setQuery(query)
            self.sparql.setTimeout(300)  # Increase timeout to 5 minutes
            
            # Convert query result to JSON format
            results = self.sparql.query().convert()
            
            # Process results and group by property
            property_data = {}
            for result in results["results"]["bindings"]:
                try:
                    prop_id = result["property"]["value"].split("/")[-1]
                    label = result["propertyLabel"]["value"]
                    
                    if prop_id not in property_data:
                        property_data[prop_id] = {
                            "label": label,
                            "aliases": set()
                        }
                    
                    if "altLabel" in result:
                        property_data[prop_id]["aliases"].add(result["altLabel"]["value"])
                except Exception as e:
                    self.logger.warning(f"Skipping malformed property: {str(e)}")
                    continue
            
            self.logger.info("✨ Computing embeddings for labels and aliases...")
            # Generate embeddings for labels and aliases
            for prop_id, data in property_data.items():
                try:
                    # Convert aliases set to list for JSON serialization
                    aliases = list(data["aliases"])
                    
                    # Generate embedding for main label
                    label_embedding = self.embedding_model.encode(data["label"]).tolist()
                    
                    # Generate embeddings for aliases
                    alias_embeddings = {}
                    for alias in aliases:
                        alias_embedding = self.embedding_model.encode(alias).tolist()
                        alias_embeddings[alias] = alias_embedding
                    
                    # Store all information
                    properties[prop_id] = {
                        "label": data["label"],
                        "label_embedding": label_embedding,
                        "aliases": aliases,
                        "alias_embeddings": alias_embeddings
                    }
                except Exception as e:
                    self.logger.warning(f"Error processing property {prop_id}: {str(e)}")
                    continue
            
            self.logger.info(f"📝 Saving properties to {output_file}...")
            try:
                # First try saving with pretty printing
                with open(output_file, "w", encoding='utf-8') as file:
                    json.dump(properties, file, indent=4, ensure_ascii=False)
            except Exception as e:
                self.logger.warning(f"Failed to save with pretty printing: {str(e)}")
                # If that fails, try without pretty printing
                with open(output_file, "w", encoding='utf-8') as file:
                    json.dump(properties, file, ensure_ascii=False)
            
            self.logger.info(f"✅ Successfully saved {len(properties)} properties with their aliases and embeddings")
            self.logger.info(f"📊 Statistics:")
            total_aliases = sum(len(p["aliases"]) for p in properties.values())
            self.logger.info(f"   • Total properties: {len(properties)}")
            self.logger.info(f"   • Total aliases: {total_aliases}")
            self.logger.info(f"   • Average aliases per property: {total_aliases/len(properties):.2f}")
            
            return properties
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching properties: {str(e)}")
            raise

    def load_embeddings(self, file_path: str, use_aliases: bool = True) -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
        """
        Load stored embeddings from a file and convert to appropriate format if needed.
        
        Args:
            file_path: Path to the embeddings file
            use_aliases: Whether to use the alias format
            
        Returns:
            Dictionary containing property information with embeddings
        """
        try:
            with open(file_path, "r") as file:
                properties = json.load(file)
            
            self.logger.info(f"✅ Successfully loaded properties from {file_path}")
            
            # Check if we need to convert from old format to new format
            if use_aliases:
                # Check if the first property has the new format structure
                first_prop = next(iter(properties.values()))
                if "label_embedding" not in first_prop:
                    self.logger.info("🔄 Converting properties to alias format...")
                    converted_properties = {}
                    for prop_id, prop_data in properties.items():
                        converted_properties[prop_id] = {
                            "label": prop_data["label"],
                            "label_embedding": prop_data["embedding"],
                            "aliases": [],
                            "alias_embeddings": {}
                        }
                    properties = converted_properties
                    self.logger.info("✅ Properties converted to alias format.")
            else:
                # Check if we need to convert from new format to old format
                first_prop = next(iter(properties.values()))
                if "embedding" not in first_prop:
                    self.logger.info("🔄 Converting properties to simple format...")
                    converted_properties = {}
                    for prop_id, prop_data in properties.items():
                        converted_properties[prop_id] = {
                            "label": prop_data["label"],
                            "embedding": prop_data["label_embedding"]
                        }
                    properties = converted_properties
                    self.logger.info("✅ Properties converted to simple format.")
            
            return properties
        except Exception as e:
            self.logger.error(f"❌ Error loading embeddings: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    generator = WikidataEmbeddingGenerator(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings without aliases
    properties = generator.generate_embeddings("wikidata-properties.json", use_aliases=False)
    
    # Generate embeddings with aliases
    properties_with_aliases = generator.generate_embeddings("wikidata-properties-with-aliases.json", use_aliases=True) 