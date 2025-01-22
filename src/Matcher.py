from typing import List, Dict, Union
import os
from urllib.parse import quote
import torch
from sentence_transformers import util
import requests
from time import sleep
from logger_config import setup_logger

class Matcher:
    """
    Class for finding and saving matches between predicates and Wikidata properties.
    Also handles entity and property matching with Wikidata.
    """
    def __init__(self, embedding_model, properties: Dict, matching_config: Dict = None):
        """
        Initialize the Matcher.
        
        Args:
            embedding_model: The initialized sentence transformer model
            properties: Dictionary of properties with their embeddings
            matching_config: Configuration for matching method
        """
        self.logger = setup_logger(__name__)
        self.embedding_model = embedding_model
        self.properties = properties
        self.matching_config = matching_config
        self.entity_cache = {}
        self.property_cache = {}

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
            self.logger.error(f"Error searching for entity {label}: {e}")
            return None

    def query_wikidata_property(self, predicate: str, language: str = "en") -> str:
        """
        Search for Wikidata properties using either Best_match or the Wikidata API.
        
        Args:
            predicate: The predicate text to search for
            method: "cos_similarity", "api", or "mixed" for the search method
                   (mixed tries api first, then cos_similarity if no exact match)
            language: Language code (default: "en")
        
        Returns:
            Property ID or None if not found
        """
        if self.matching_config.get("property_query_method").lower() in ["api", "mixed"]:
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
                            self.logger.info(f"Found exact match for '{predicate}': {property_id} ({result.get('label')})")
                            return property_id
                    
                    # If no exact match found and method is "api", take the first result
                    if self.matching_config.get("property_query_method").lower() == "api":
                        property_id = data["search"][0]["id"]
                        label = data["search"][0].get("label", "")
                        self.property_cache[predicate] = property_id
                        self.logger.info(f"Found closest match for '{predicate}': {property_id} ({label})")
                        return property_id
                    
                    # If method is "mixed" and no exact match, try cos_similarity
                    elif self.matching_config.get("property_query_method").lower() == "mixed":
                        self.logger.info(f"No exact API match for '{predicate}', trying cosine similarity...")
                        return self.find_best_match(predicate)
                
                elif self.matching_config.get("property_query_method").lower() == "mixed":
                    self.logger.info(f"No API results for '{predicate}', trying cosine similarity...")
                    return self.find_best_match(predicate)
                
                self.logger.info(f"No property found for '{predicate}'")
                return None
                
            except Exception as e:
                self.logger.error(f"Error searching for property '{predicate}': {e}")
                if self.matching_config.get("property_query_method").lower() == "mixed":
                    self.logger.info(f"Trying cosine similarity as fallback...")
                    return self.find_best_match(predicate)
                return None
        
        elif self.matching_config.get("property_query_method").lower() == "cos_similarity":
            return self.find_best_match(predicate)
        
        else:
            raise ValueError(f"Invalid method: {self.matching_config.get('property_query_method')}. Use 'api', 'cos_similarity', or 'mixed'.")

    def find_best_match(self, predicate: str):
        """
        Find and save top 10 matches for a predicate based on cosine similarity.
        Uses either simple matching or matching with aliases based on configuration.
        """
        if not self.properties:
            self.logger.warning("No properties loaded. Please ensure properties are initialized.")
            return None

        if self.matching_config.get("use_aliases", True):
            return self._find_best_match_with_aliases(predicate)
        else:
            return self._find_best_match_simple(predicate)

    def _find_best_match_simple(self, predicate: str):
        """Simple matching using only main labels."""
        predicate_embedding = self.embedding_model.encode(predicate)
        matches = []
        
        for prop_id, prop_data in self.properties.items():
            similarity = util.cos_sim(predicate_embedding, torch.tensor(prop_data["embedding"])).item()
            matches.append({
                "property_id": prop_id,
                "label": prop_data["label"],
                "similarity": similarity
            })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = matches[:10]
        self.save_property_matches(predicate, top_matches)
        
        if top_matches:
            best_match = top_matches[0]
            self.logger.info(f"\n🎯 Best Match for '{predicate}':")
            self.logger.info(f"🔑 Property ID: {best_match['property_id']}")
            self.logger.info(f"📝 Label: {best_match['label']}")
            self.logger.info(f"💯 Similarity: {best_match['similarity']:.4f}")
            return best_match["property_id"]
        return None

    def _find_best_match_with_aliases(self, predicate: str):
        """Advanced matching using both labels and aliases."""
        predicate_embedding = self.embedding_model.encode(predicate)
        matches = []
        
        for prop_id, prop_data in self.properties.items():
            label_similarity = util.cos_sim(predicate_embedding, torch.tensor(prop_data["label_embedding"])).item()
            
            alias_similarities = []
            for alias in prop_data["aliases"]:
                alias_embedding = prop_data["alias_embeddings"][alias]
                similarity = util.cos_sim(predicate_embedding, torch.tensor(alias_embedding)).item()
                alias_similarities.append(similarity)
            
            best_similarity = max([label_similarity] + alias_similarities) if alias_similarities else label_similarity
            best_match_text = prop_data["label"]
            
            if alias_similarities:
                max_alias_idx = alias_similarities.index(max(alias_similarities))
                if alias_similarities[max_alias_idx] > label_similarity:
                    best_match_text = prop_data["aliases"][max_alias_idx]
            
            matches.append({
                "property_id": prop_id,
                "label": prop_data["label"],
                "best_match_text": best_match_text,
                "similarity": best_similarity
            })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = matches[:10]
        self.save_property_matches(predicate, top_matches)
        
        if top_matches:
            best_match = top_matches[0]
            self.logger.info(f"\n🎯 Best Match for '{predicate}':")
            self.logger.info(f"🔑 Property ID: {best_match['property_id']}")
            self.logger.info(f"📝 Label: {best_match['label']}")
            if best_match['best_match_text'] != best_match['label']:
                self.logger.info(f"✨ Matched via alias: {best_match['best_match_text']}")
            self.logger.info(f"💯 Similarity: {best_match['similarity']:.4f}")
            return best_match["property_id"]
        return None

    def save_property_matches(self, predicate: str, matches: List[Dict], output_dir: str = "property_matches"):
        """
        Save top property matches to a file.
        
        Args:
            predicate: The predicate that was searched for
            matches: List of dictionaries containing match information
            output_dir: Directory to save the matches file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a filename from the predicate
            safe_filename = quote(predicate.replace(" ", "_"))
            output_file = os.path.join(output_dir, f"{safe_filename}_matches.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Top matches for predicate: \"{predicate}\"\n")
                f.write("="*50 + "\n\n")
                
                for i, match in enumerate(matches, 1):
                    f.write(f"{i}. Match Details:\n")
                    f.write(f"   Property ID: {match['property_id']}\n")
                    f.write(f"   Label: {match['label']}\n")
                    f.write(f"   Similarity Score: {match['similarity']:.4f}\n")
                    f.write("\n")
                    
            self.logger.info(f"✓ Saved property matches to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"⚠️  Error saving property matches: {str(e)}") 