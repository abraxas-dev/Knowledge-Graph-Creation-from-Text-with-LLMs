from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from typing import Dict, Tuple
from urllib.parse import quote
from .logger_config import setup_logger

class GraphManager:
    """
    Class for managing RDF graph operations and statistics.
    Handles graph initialization, triple addition, and graph statistics.
    """
    def __init__(self):
        """Initialize the RDF graph and set up Wikidata namespaces."""
        self.logger = setup_logger(__name__)
        self.g = Graph()
        self.wd = Namespace("http://www.wikidata.org/entity/")
        self.wdt = Namespace("http://www.wikidata.org/prop/direct/")
        
        # Bind namespaces
        self.g.bind("wd", self.wd)
        self.g.bind("wdt", self.wdt)
        
        self.logger.info("✅ Successfully initialized knowledge graph and namespaces")

    def add_triple(self, subject_id: str, property_id: str, object_id: str, is_literal: bool = False) -> None:
        """
        Add a triple to the graph.
        
        Args:
            subject_id: Wikidata ID for subject or raw subject text
            property_id: Wikidata property ID or raw predicate text
            object_id: Wikidata ID for object or raw object text
            is_literal: Whether the object should be treated as a literal value
        """
        subject_uri = self.wd[subject_id] if subject_id.startswith('Q') else URIRef(f"http://example.org/entity/{quote(subject_id)}")
        property_uri = self.wdt[property_id] if property_id.startswith('P') else URIRef(f"http://example.org/property/{quote(property_id.lower().replace(' ', '_'))}")
        
        if is_literal:
            self.g.add((subject_uri, property_uri, Literal(object_id)))
        else:
            object_uri = self.wd[object_id] if object_id.startswith('Q') else URIRef(f"http://example.org/entity/{quote(object_id)}")
            self.g.add((subject_uri, property_uri, object_uri))

    def get_statistics(self) -> Dict:
        """
        Get statistics about the current graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        return {
            "total_triples": len(self.g),
            "unique_subjects": len(set(self.g.subjects())),
            "unique_predicates": len(set(self.g.predicates())),
            "unique_objects": len(set(self.g.objects()))
        }

    def print_statistics(self, title: str = "Current Statistics") -> None:
        """
        Print formatted statistics about the current graph.
        
        Args:
            title: Title for the statistics section
        """
        stats = self.get_statistics()
        
        self.logger.info("="*50)
        self.logger.info(f"📊 {title}")
        self.logger.info("="*50)
 
        max_num_width = max(len(str(v)) for v in stats.values())
        
        self.logger.info(f"🔸 Total Triples:       {stats['total_triples']:>{max_num_width}}")
        self.logger.info(f"🔸 Unique Subjects:     {stats['unique_subjects']:>{max_num_width}}")
        self.logger.info(f"🔸 Unique Predicates:   {stats['unique_predicates']:>{max_num_width}}")
        self.logger.info(f"🔸 Unique Objects:      {stats['unique_objects']:>{max_num_width}}\n")
        
        self.logger.info("="*50)

    def save_graph(self, output_file: str, format: str = "turtle") -> None:
        """
        Save the graph to a file.
        
        Args:
            output_file: Path to save the graph
            format: Format to save the graph in (default: turtle)
        """
        try:
            self.g.serialize(destination=output_file, format=format)
            self.logger.info(f"✅ Successfully saved graph to {output_file}")
        except Exception as e:
            self.logger.error(f"❌ Error saving graph: {str(e)}")
            raise

    def load_graph(self, input_file: str, format: str = "turtle") -> None:
        """
        Load a graph from a file.
        
        Args:
            input_file: Path to the graph file
            format: Format of the graph file (default: turtle)
        """
        try:
            self.g.parse(input_file, format=format)
            self.logger.info(f"✅ Successfully loaded graph from {input_file}")
        except Exception as e:
            self.logger.error(f"❌ Error loading graph: {str(e)}")
            raise 