import os
import json
from .extractor import Extractor
from .triple_generator import TripleGenerator
from .Integrator import Integrator

class Pipeline:
    """
    Pipeline class that orchestrates the complete knowledge graph extraction process:
    1. Extract and preprocess content from webpages
    2. Generate triples from extracted content
    3. Integrate triples with Wikidata/DBpedia
    """

    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # Create necessary directories
        os.makedirs(self.config["data_paths"]["processed_data_path"], exist_ok=True)
        os.makedirs(self.config["data_paths"]["triples_path"], exist_ok=True)
        os.makedirs(self.config["data_paths"]["knowledge_graph_path"], exist_ok=True)
        
        # Initialize extractor
        self.extractor = Extractor(
            urls=self.config["extractor"]["urls"],
            processed_data_path=self.config["data_paths"]["processed_data_path"],
            chunk_size=self.config["extractor"]["chunk_size"]
        )

        # Initialize triple generator
        self.triple_generator = TripleGenerator(
            api_key=self.config["llm"]["api_key"],
            input_dir=self.config["data_paths"]["processed_data_path"],
            output_dir=self.config["data_paths"]["triples_path"],
            system_message=self.config["llm"]["system_message"],
            prompt_template=self.config["llm"]["prompt_template"],
            model_name=self.config["llm"]["model_name"],
            max_new_tokens=self.config["llm"]["max_new_tokens"],
            temperature=self.config["llm"]["temperature"],
            max_chunks=self.config["llm"].get("max_chunks", None)
        )

        # Initialize integrator
        self.integrator = Integrator(
            input_dir=self.config["data_paths"]["triples_path"],
            output_dir=self.config["data_paths"]["knowledge_graph_path"],
            embedding_model=self.config["integrator"]["embedding_model"],
            matching_config=self.config["integrator"]["matching_method"]
        )

    def run(self):
        """Execute the complete pipeline."""
        try:
            print("\n" + "="*50)
            print("Pipeline execution started.")
            print("="*50)

            # Step 1: Extract and preprocess content
            print("\nStep 1: Extracting and preprocessing webpage content...")
            print("-"*50)
            self.extractor.preprocess()
            print("Content extraction and preprocessing completed.")

            # Step 2: Generate triples
            print("\nStep 2: Generating knowledge graph triples...")
            print("-"*50)
            self.triple_generator.process()
            print("Triple generation completed.")

            # Step 3: Integrate with knowledge bases
            print("\nStep 3: Integrating triples with knowledge bases...")
            print("-"*50)
            self.integrator.process_directory()
            
            # Save final knowledge graph
            output_file = os.path.join(
                self.config["data_paths"]["knowledge_graph_path"], 
                "knowledge_graph.ttl"
            )
            self.integrator.save_graph(output_file)
            
            # Get and print statistics
            stats = self.integrator.get_statistics()
            print("\nKnowledge Graph Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
            print("Integration completed.")

            print("\n" + "="*50)
            print("Pipeline execution completed successfully!")
            print("="*50 + "\n")

        except Exception as e:
            print("\n" + "="*50)
            print(f"Pipeline execution failed: {str(e)}")
            print("="*50 + "\n")
            raise

if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    pipeline = Pipeline(CONFIG_PATH)
    pipeline.run()
