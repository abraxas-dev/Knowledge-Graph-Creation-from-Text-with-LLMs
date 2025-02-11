"""
Project : Knowledge Graph Creation from Text
Author : @abraxas-dev
"""
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from typing import Dict, Tuple
from urllib.parse import quote
from src.utils.logger_config import setup_logger
import networkx as nx
import matplotlib.pyplot as plt
import re
from pathlib import Path


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
        
        # Initialize NetworkX graph for visualization
        self.nx_graph = nx.DiGraph()
        self.node_labels = {}
        
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

    def _extract_id(self, uri: str) -> str:
        """
        Extract the ID or label from a URI.
        
        Args:
            uri: URI to extract ID from
            
        Returns:
            str: Extracted ID or label
        """
        wikidata_match = re.search(r'[QP]\d+$', uri)
        if wikidata_match:
            return wikidata_match.group()
        return uri.split('/')[-1].replace('_', ' ')

    def prepare_visualization(self) -> None:
        """
        Prepare the NetworkX graph for visualization by converting RDF graph.
        """
        self.nx_graph.clear()
        self.node_labels.clear()
        
        for s, p, o in self.g:
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                subject_id = self._extract_id(str(s))
                object_id = self._extract_id(str(o))
                predicate_label = self._extract_id(str(p))
                
                self.nx_graph.add_node(subject_id)
                self.nx_graph.add_node(object_id)
                self.nx_graph.add_edge(subject_id, object_id, label=predicate_label)
                
                self.node_labels[subject_id] = subject_id
                self.node_labels[object_id] = object_id

    def visualize(self, output_dir: str = "./visualizations", 
                 figsize: Tuple[int, int] = (15, 10),
                 node_size: int = 2000,
                 font_size: int = 8,
                 edge_label_font_size: int = 6,
                 title: str = "Knowledge Graph Visualization") -> None:
        """
        Create and save a visualization of the graph.
        
        Args:
            output_dir: Directory to save the visualization
            figsize: Size of the figure (width, height)
            node_size: Size of nodes in the visualization
            font_size: Font size for node labels
            edge_label_font_size: Font size for edge labels
            title: Title of the visualization
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare the graph for visualization
            self.prepare_visualization()
            
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                                 node_color='lightblue',
                                 node_size=node_size)
            
            # Draw edges
            nx.draw_networkx_edges(self.nx_graph, pos, 
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20)
            
            # Draw node labels
            nx.draw_networkx_labels(self.nx_graph, pos,
                                  labels=self.node_labels,
                                  font_size=font_size)
            
            # Draw edge labels
            edge_labels = nx.get_edge_attributes(self.nx_graph, 'label')
            nx.draw_networkx_edge_labels(self.nx_graph, pos,
                                       edge_labels=edge_labels,
                                       font_size=edge_label_font_size)
            
            plt.title(title)
            plt.axis('off')
            
            # Save the visualization
            output_file = output_dir / "knowledge_graph_visualization.png"
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Saved visualization to {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating visualization: {str(e)}")
            raise 