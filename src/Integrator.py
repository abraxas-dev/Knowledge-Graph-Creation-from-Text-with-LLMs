import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from typing import List, Tuple, Dict
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from time import sleep
from urllib.parse import quote

class Integrator:
    def __init__(self):
        self.g = Graph()
        self.wd = Namespace("http://www.wikidata.org/entity/")
        self.wdt = Namespace("http://www.wikidata.org/prop/direct/")
        self.g.bind("wd", self.wd)
        self.g.bind("wdt", self.wdt)
        
        self.entity_cache = {}
        self.property_cache = {}

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

    def query_wikidata_property(self, predicate: str, language: str = "en") -> str:
        """
        Sucht nach einer Wikidata-Property basierend auf einem Prädikat
        """
        if predicate in self.property_cache:
            return self.property_cache[predicate]

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = f"""
        SELECT ?property ?propertyLabel WHERE {{
            ?property wikibase:propertyType wikibase:WikibaseProperty;
                     rdfs:label ?propertyLabel.
            FILTER(LANG(?propertyLabel) = "{language}")
            FILTER(CONTAINS(LCASE(?propertyLabel), LCASE("{predicate}")))
        }}
        LIMIT 1
        """
        
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            if results["results"]["bindings"]:
                property_id = results["results"]["bindings"][0]["property"]["value"].split("/")[-1]
                self.property_cache[predicate] = property_id
                return property_id
            return None
        except Exception as e:
            print(f"Fehler beim Suchen der Property {predicate}: {e}")
            return None

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
    pipeline.process_triples(example_triples)
    
    stats = pipeline.get_statistics()
    print("Statistiken:", json.dumps(stats, indent=2))
    
    pipeline.save_graph("wikidata_knowledge_graph.ttl")