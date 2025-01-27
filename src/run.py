"""
Project : Knowledge Graph Creation from Text
Author : @abraxas-dev
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import argparse
import yaml
from typing import Dict
from src.core.Extractor import Extractor
from src.core.TripleGenerator import TripleGenerator
from src.core.Integrator.Integrator import Integrator
from src.utils.logger_config import setup_logger   

logger = setup_logger(__name__)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        sys.exit(1)

def setup_directories(config: Dict) -> None:
    """Create necessary directories from config."""
    try:
        for path in config["data_paths"].values():
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ Ensured directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        sys.exit(1)

def run_pipeline(config: Dict, mode: str = "full") -> None:
    """
    Run the knowledge graph generation pipeline.
    
    Args:
        config: Configuration dictionary
        mode: Pipeline execution mode ('full', 'extract', 'generate', or 'integrate')
    """
    try:
        logger.info("="*50)
        logger.info(f"Starting pipeline in {mode} mode")
        logger.info("="*50)

        if mode in ["full", "extract"]:
            logger.info("📑 Starting content extraction...")
            extractor = Extractor(
                urls=config["extractor"]["urls"],
                processed_data_path=config["data_paths"]["processed_data_path"],
                chunk_size=config["extractor"]["chunk_size"]
            )
            extractor.preprocess()

        if mode in ["full", "generate"]:
            logger.info("🔄 Starting triple generation...")
            generator = TripleGenerator(
                api_key=config["llm"]["api_key"],
                input_dir=config["data_paths"]["processed_data_path"],
                output_dir=config["data_paths"]["triples_path"],
                system_message=config["llm"]["system_message"],
                prompt_template=config["llm"]["prompt_template"],
                model_name=config["llm"]["model_name"],
                max_chunks=config["llm"].get("max_chunks"),
                model_generate_parameters=config["llm"].get("model_generate_parameters")
            )
            generator.process()

        if mode in ["full", "integrate"]:
            logger.info("🔗 Starting knowledge integration...")
            integrator = Integrator(
                input_dir=config["data_paths"]["triples_path"],
                output_dir=config["data_paths"]["knowledge_graph_path"],
                embedding_model=config["integrator"]["embedding_model"],
                matching_config=config["integrator"]["matching_method"]
            )
            integrator.process()

        logger.info("="*50)
        logger.info("✅ Pipeline execution completed successfully!")
        logger.info("="*50)

    except Exception as e:
        logger.error("="*50)
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        logger.error("="*50)
        raise

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        default="./config/main.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "extract", "generate", "integrate"],
        default="full",
        help="Pipeline execution mode (default: full)"
    )

    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    setup_directories(config)
    
    # Run pipeline
    run_pipeline(config, args.mode)

if __name__ == "__main__":
    main() 