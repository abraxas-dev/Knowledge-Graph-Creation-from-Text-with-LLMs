"""
Knowledge Graph Generation Pipeline

This module serves as the main entry point for the Knowledge Graph Generation pipeline.
It orchestrates the three main components: Extractor, TripleGenerator, and Integrator.

The pipeline can be run in different modes:
- full: Executes all components sequentially
- extract: Only runs the content extraction phase
- generate: Only runs the triple generation phase
- integrate: Only runs the knowledge graph integration phase

Author: @abraxas-dev
"""
import os
import sys
from pathlib import Path
import argparse
import yaml
from typing import Dict
from src.core.Extractor import Extractor
from core.Generator.TripleGenerator import TripleGenerator
from src.core.Integrator.Integrator import Integrator
from src.utils.logger_config import setup_logger   

# Configure Python path to include project root for proper imports
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

logger = setup_logger(__name__)

def load_config(config_path: str) -> Dict:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        Dict: Parsed configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        sys.exit(1)

def setup_directories(config: Dict) -> None:
    """
    Create all necessary directories specified in the configuration.
    
    This ensures that all required directories exist before pipeline execution.
    Creates directories for processed data, generated triples, and the final
    knowledge graph.

    Args:
        config (Dict): Configuration dictionary containing data paths

    Raises:
        SystemExit: If directory creation fails
    """
    try:
        for path in config["data_paths"].values():
            os.makedirs(path, exist_ok=True)
            logger.info(f"✅ Ensured directory exists: {path}")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        sys.exit(1)

def run_pipeline(config: Dict, mode: str = "full") -> None:
    """
    This function orchestrates the three main components of the pipeline:
    1. Extractor: Extracts content from specified URLs
    2. TripleGenerator: Generates RDF triples from extracted content
    3. Integrator: Integrates generated triples into a knowledge graph

    Args:
        config (Dict): Configuration dictionary containing all necessary parameters
        mode (str): Pipeline execution mode. Options:
                   - "full": Run all components
                   - "extract": Only run content extraction
                   - "generate": Only run triple generation
                   - "integrate": Only run knowledge graph integration
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
    """
    Main entry point for the Knowledge Graph Generation pipeline.

    Handles command-line argument parsing and initiates the pipeline execution.
    Supports different execution modes and custom configuration files.

    Command-line Arguments:
        --config: Path to the configuration file (default: ./config/main.yaml)
        --mode: Pipeline execution mode (default: full)
               Options: full, extractor, generator, integrator
    """
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        default="./config/main.yaml",
        help="Path to configuration file (default: main.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "extractor", "generator", "integrator"],
        default="full",
        help="Pipeline execution mode (default: full)"
    )

    args = parser.parse_args()
    
    config = load_config(args.config)
    
    setup_directories(config)

    run_pipeline(config, args.mode)

if __name__ == "__main__":
    main() 