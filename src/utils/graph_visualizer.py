import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef
from typing import Dict, Set, Tuple
import re
from pathlib import Path
from src.utils.logger_config import setup_logger

class GraphVisualizer:
    """Class for visualizing RDF knowledge graphs."""
    
    def __init__(self, input_file: str, output_dir: str = "./visualizations"):
        """
        Initialize the graph visualizer.
        """
        self.logger = setup_logger(__name__)
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rdf_graph = Graph()
        self.nx_graph = nx.DiGraph()
        
        self.node_labels: Dict[str, str] = {}
    
    def _extract_id(self, uri: str) -> str:
        """Extract the ID or label from a URI."""

        wikidata_match = re.search(r'[QP]\d+$', uri)
        if wikidata_match:
            return wikidata_match.group()
        
        return uri.split('/')[-1].replace('_', ' ')
    
    def load_graph(self) -> None:
        """Load the RDF graph from the turtle file."""

        try:
            self.logger.info(f"Loading graph from {self.input_file}")
            self.rdf_graph.parse(self.input_file, format="turtle")
            
            # Convert RDF graph to NetworkX graph
            for s, p, o in self.rdf_graph:
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    subject_id = self._extract_id(str(s))
                    object_id = self._extract_id(str(o))
                    predicate_label = self._extract_id(str(p))
                    
                    self.nx_graph.add_node(subject_id)
                    self.nx_graph.add_node(object_id)
                    self.nx_graph.add_edge(subject_id, object_id, label=predicate_label)
                    
                    self.node_labels[subject_id] = subject_id
                    self.node_labels[object_id] = object_id
            
            self.logger.info(f"✅ Loaded graph with {len(self.nx_graph.nodes)} nodes and {len(self.nx_graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading graph: {str(e)}")
            raise
    
    def visualize(self, figsize: Tuple[int, int] = (15, 10), 
                 node_size: int = 2000,
                 font_size: int = 8,
                 edge_label_font_size: int = 6,
                 title: str = "Knowledge Graph Visualization") -> None:
        """
        Create and save a visualization of the graph.
        
        Args:
            figsize: Size of the figure (width, height)
            node_size: Size of nodes in the visualization
            font_size: Font size for node labels
            edge_label_font_size: Font size for edge labels
            title: Title of the visualization
        """

        try:
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(self.nx_graph, k=1, iterations=50)
            nx.draw_networkx_nodes(self.nx_graph, pos, 
                                 node_color='lightblue',
                                 node_size=node_size)
            nx.draw_networkx_edges(self.nx_graph, pos, 
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20)
            nx.draw_networkx_labels(self.nx_graph, pos,
                                  labels=self.node_labels,
                                  font_size=font_size)
            edge_labels = nx.get_edge_attributes(self.nx_graph, 'label')
            nx.draw_networkx_edge_labels(self.nx_graph, pos,
                                       edge_labels=edge_labels,
                                       font_size=edge_label_font_size)
            
            plt.title(title)
            plt.axis('off')

            output_file = self.output_dir / "knowledge_graph_visualization.png"
            plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Saved visualization to {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating visualization: {str(e)}")
            raise

if __name__ == "__main__":
    
    visualizer = GraphVisualizer(
        input_file="./data/run/knowledge_graph/knowledge_graph.ttl",
        output_dir="./visualizations"
    )
    visualizer.load_graph()
    visualizer.visualize() 