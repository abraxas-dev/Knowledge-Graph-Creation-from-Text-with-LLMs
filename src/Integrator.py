import os
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Tuple

class TripleIntegrator:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_triples_from_file(self, file_path: Path):
        """Load triples from a single file."""
        triples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        subject, predicate, obj = line.strip().split('\t')
                        triples.append((subject, predicate, obj))
            return triples
        except Exception as e:
            print(f"Error loading triples from {file_path}: {str(e)}")
            return []

    def get_dbpedia_uri(self, label: str):
        """Query DBpedia for URI of a given label."""
        try:
            self.sparql.setQuery(f'''
                SELECT ?entity WHERE {{
                    ?entity rdfs:label "{label}"@en.
                }}
            ''')
            results = self.sparql.query().convert()
            bindings = results.get("results", {}).get("bindings", [])
            
            return bindings[0]["entity"]["value"] if bindings else None
        except Exception as e:
            print(f"Error querying DBpedia for '{label}': {str(e)}")
            return None

    def map_triple_to_dbpedia(self, triple: Tuple[str, str, str]):
        """Map a single triple to DBpedia URIs."""
        subject, predicate, obj = triple
        subject_resource = self.get_dbpedia_uri(subject)
        object_resource = self.get_dbpedia_uri(obj)

        if subject_resource and object_resource:
            return (subject_resource, predicate, object_resource)
        else:
            print(f"Could not map: ({subject}, {predicate}, {obj})")
            return None

    def save_mapped_triple(self, mapped_triple: Tuple[str, str, str], output_file: Path):
        """Save a single mapped triple to the output file."""
        try:
            with open(output_file, 'a', encoding='utf-8') as file:
                subject_uri, predicate, obj_uri = mapped_triple
                file.write(f"< {subject_uri} > < {predicate} > < {obj_uri} > .\n")
        except Exception as e:
            print(f"Error saving mapped triple to {output_file}: {str(e)}")

    def process_file(self, file_path: Path):
        """Process a single file of triples."""
        print(f"Processing {file_path.name}...")
        
        # Create output file for this input file
        output_file = self.output_dir / f"{file_path.stem}_mapped.ttl"
        
        # Clear output file if it exists
        if output_file.exists():
            output_file.unlink()

        # Process triples
        triples = self.load_triples_from_file(file_path)
        for triple in triples:
            mapped_triple = self.map_triple_to_dbpedia(triple)
            if mapped_triple:
                self.save_mapped_triple(mapped_triple, output_file)

    def process_directory(self):
        """Process all files in the input directory."""
        try:
            for file_path in self.input_dir.glob("*.txt"):
                self.process_file(file_path)
        except Exception as e:
            print(f"Error processing directory: {str(e)}")

def main(): 
    input_dir = "./input_triples"
    output_dir = "./output_mapped"
    
    integrator = TripleIntegrator(input_dir, output_dir)
    integrator.process_directory()

if __name__ == "__main__":
    main()
