from typing import List, Dict, Union
import os
from urllib.parse import quote
import torch
from sentence_transformers import util

class Matcher:
    """
    Class for finding and saving matches between predicates and Wikidata properties.
    """
    def __init__(self, embedding_model, properties: Dict, matching_config: Dict = None):
        """
        Initialize the Matcher.
        
        Args:
            embedding_model: The initialized sentence transformer model
            properties: Dictionary of properties with their embeddings
            matching_config: Configuration for matching method
        """
        self.embedding_model = embedding_model
        self.properties = properties
        self.matching_config = matching_config or {
            "use_aliases": True,
            "properties_file": "wikidata-properties-with-aliases.json"
        }

    def find_best_match(self, predicate: str):
        """
        Find and save top 10 matches for a predicate based on cosine similarity.
        Uses either simple matching or matching with aliases based on configuration.
        """
        if not self.properties:
            print("No properties loaded. Please ensure properties are initialized.")
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
            print(f"\n🎯 Best Match for '{predicate}':")
            print(f"🔑 Property ID: {best_match['property_id']}")
            print(f"📝 Label: {best_match['label']}")
            print(f"💯 Similarity: {best_match['similarity']:.4f}")
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
            print(f"\n🎯 Best Match for '{predicate}':")
            print(f"🔑 Property ID: {best_match['property_id']}")
            print(f"📝 Label: {best_match['label']}")
            if best_match['best_match_text'] != best_match['label']:
                print(f"✨ Matched via alias: {best_match['best_match_text']}")
            print(f"💯 Similarity: {best_match['similarity']:.4f}")
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
                    
            print(f"✓ Saved property matches to: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Error saving property matches: {str(e)}") 