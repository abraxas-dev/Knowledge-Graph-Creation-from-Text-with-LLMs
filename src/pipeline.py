import os
import json
from extractor import Extractor
from triple_generator import TripleGenerator

class Pipeline:

    def __init__(self, config_path):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        
        self.extractor = Extractor(
            raw_data_path=self.config["extractor"]["urls"],
            processed_data_path=self.config["data_paths"]["processed_data_path"],
            chunk_size=self.config["extractor"]["chunk_size"]
        )
        self.triple_generator = TripleGenerator(
            api_key=self.config["llm"]["api_key"],
            model_name=self.config["llm"]["model_name"],
            input_dir=self.config["data_paths"]["processed_data_path"],
            output_dir=self.config["data_paths"]["triples_path"],
            system_message=self.config["llm"]["system_message"],
            prompt_message=self.config["llm"]["prompt_message"],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"]
        )

    def run(self):

        print("Pipeline execution started.")
        
        # Extraction
        print("Step 1: Extracting and preprocessing data...")
        self.extractor.preprocess()
    
        
        # Triple Generation
        print("Step 2: Generating triples...")
        self.triple_generator.process()
        print("Triples have been generated and saved.")


        print("Pipeline execution finished.")

if __name__ == "__main__":
    CONFIG_PATH = "config.json"
    pipeline = Pipeline(CONFIG_PATH)
    pipeline.run()
