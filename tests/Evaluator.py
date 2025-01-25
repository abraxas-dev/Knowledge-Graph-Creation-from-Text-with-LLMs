import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.core.Integrator.Integrator import Integrator
from src.core.TripleGenerator import TripleGenerator
from test_modules.metrics import TripleEvaluator
from test_modules.metrics_generator import TextTripleEvaluator
from src.utils.logger_config import setup_logger

class Evaluator:
    def __init__(self, configs_dir: str, input_dir: str, groundtruth_dir: str, output_dir: str, mode: str):
        """
        Initialize the evaluator.
        
        Args:
            configs_dir: Directory containing configuration files to test
            input_dir: Directory containing input files
            groundtruth_dir: Directory containing groundtruth files
            output_dir: Directory to save evaluation results
            mode: Evaluation mode ('integrator' or 'generator')
        """
        self.logger = setup_logger(__name__, log_dir=os.path.join(output_dir, "logs"))
        self.configs_dir = Path(configs_dir)
        self.input_dir = Path(input_dir)
        self.groundtruth_dir = Path(groundtruth_dir)
        self.output_dir = Path(output_dir)
        self.mode = mode
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if mode == 'integrator':
            # Create Properties directory if it doesn't exist
            properties_dir = os.path.join(project_root, "src/data/Properties")
            os.makedirs(properties_dir, exist_ok=True)
        
    def _load_config(self, config_path: Path) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
            
    def _get_config_files(self) -> List[Path]:
        return list(self.configs_dir.glob("*.yaml"))

    def evaluate_integrator_configuration(self, config_path: Path) -> Dict:
        """Evaluate a single configuration of the Integrator."""
        try:
            config = self._load_config(config_path)
            
            config_name = config_path.stem
            config_output_dir = self.output_dir / config_name
            config_output_dir.mkdir(exist_ok=True)
            
            # Make properties file path absolute
            properties_file = config.get('properties_file', 'src/data/Properties/wikidata-properties-with-aliases.json')
            properties_file = os.path.join(project_root, properties_file)
            
            # Create property_matches directory in the config output directory
            property_matches_dir = os.path.join(config_output_dir, "property_matches")
            os.makedirs(property_matches_dir, exist_ok=True)
            
            integrator = Integrator(
                input_dir=str(self.input_dir),
                output_dir=str(config_output_dir),
                embedding_model=config.get('embedding_model', "sentence-transformers/all-mpnet-base-v2"),
                matching_config={
                    "use_aliases": config.get('use_aliases', True),
                    "properties_file": properties_file,
                    "entity_query_method": config.get('entity_query_method', "api"),
                    "property_query_method": config.get('property_query_method', "api"),
                    "save_matches": config.get('save_matches', True),
                    "matches_output_file": "matched_triples.yaml",
                    "property_matches_dir": property_matches_dir
                }
            )
            
            integrator.process()
            
            # Evaluate results using TripleEvaluator
            matched_triples_file = config_output_dir / "matched_triples.yaml"
            groundtruth_file = self.groundtruth_dir / "groundtruth_triples.yaml"
            
            if not matched_triples_file.exists():
                raise FileNotFoundError(f"Integrator did not generate matched_triples.yaml in {config_output_dir}")
            if not groundtruth_file.exists():
                raise FileNotFoundError(f"Ground truth file not found: {groundtruth_file}")
            
            evaluator = TripleEvaluator(
                gen_file=str(matched_triples_file),
                gt_file=str(groundtruth_file)
            )
            
            metrics = evaluator.evaluate()
            
            # Save metrics
            metrics_file = config_output_dir / "metrics.yaml"
            with open(metrics_file, 'w') as f:
                yaml.dump({
                    'configuration': config,
                    'metrics': metrics
                }, f, default_flow_style=False)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating configuration {config_path}: {e}")
            return None

    def evaluate_generator_configuration(self, config_path: Path) -> Dict:
        """Evaluate a single configuration of the TripleGenerator."""
        try:
            config = self._load_config(config_path)
            
            config_name = config_path.stem
            config_output_dir = self.output_dir / config_name
            config_output_dir.mkdir(exist_ok=True)
            
            generator = TripleGenerator(
                api_key=config.get('api_key', ''),
                input_dir=str(config.get('input_dir', self.input_dir)),
                output_dir=str(config_output_dir),
                system_message=config.get('system_message', ''),
                prompt_template=config.get('prompt_template', ''),
                model_name=config.get('model_name', ''),
                max_chunks=config.get('max_chunks'),
                model_generate_parameters=config.get('model_generate_parameters')
            )
            
            generator.process()
            
            # Find valid and invalid triples files
            valid_file = config_output_dir / "all_valid_triples.txt"
            invalid_file = config_output_dir / "all_invalid_triples.txt"
            groundtruth_file = self.groundtruth_dir / "groundtruth_triples.txt"
            
            if not valid_file.exists() or not invalid_file.exists():
                raise FileNotFoundError(f"Generator did not produce expected output files in {config_output_dir}")
            if not groundtruth_file.exists():
                raise FileNotFoundError(f"Ground truth file not found: {groundtruth_file}")
            
            evaluator = TextTripleEvaluator(
                valid_file=str(valid_file),
                invalid_file=str(invalid_file),
                gt_file=str(groundtruth_file)
            )
            
            metrics = evaluator.evaluate()
            
            # Save metrics
            metrics_file = config_output_dir / "metrics.yaml"
            with open(metrics_file, 'w') as f:
                yaml.dump({
                    'configuration': config,
                    'metrics': metrics
                }, f, default_flow_style=False)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating configuration {config_path}: {e}")
            return None
            
    def run_evaluation(self):
        """Run evaluation for all configuration files."""
        try:
            self.logger.info("="*50)
            self.logger.info(f"Starting {self.mode.capitalize()} Evaluation")
            self.logger.info("="*50)
            
            config_files = self._get_config_files()
            if not config_files:
                self.logger.warning(f"No configuration files found in {self.configs_dir}")
                return
                
            self.logger.info(f"Found {len(config_files)} configuration files to test")
            
            all_results = []
            
            for i, config_file in enumerate(config_files, 1):
                self.logger.info(f"\nEvaluating configuration {i}/{len(config_files)}")
                self.logger.info(f"Configuration file: {config_file.name}")
                
                if self.mode == 'integrator':
                    metrics = self.evaluate_integrator_configuration(config_file)
                else:
                    metrics = self.evaluate_generator_configuration(config_file)
                
                if metrics:
                    all_results.append({
                        'configuration_file': config_file.name,
                        'metrics': metrics
                    })
                    
                    self.logger.info(f"Results for configuration {config_file.name}:")
                    for category, values in metrics.items():
                        self.logger.info(f"\n{category}:")
                        for metric, value in values.items():
                            if isinstance(value, (int, float)):
                                self.logger.info(f"  {metric}: {value:.4f}")
                            else:
                                self.logger.info(f"  {metric}: {value}")
            
            # Save overall results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"evaluation_results_{timestamp}.yaml"
            
            with open(results_file, 'w') as f:
                yaml.dump({
                    'evaluation_timestamp': timestamp,
                    'results': all_results
                }, f, default_flow_style=False)
                
            self.logger.info("\n" + "="*50)
            self.logger.info(f"Evaluation complete. Results saved to {results_file}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Evaluate Integrator or TripleGenerator')
    parser.add_argument('--mode', choices=['integrator', 'generator'], required=True,
                      help='Evaluation mode: integrator or generator')
    args = parser.parse_args()

    # Directory paths based on mode
    configs_dir = os.path.join(project_root, f"tests/config/{args.mode.capitalize()}")
    input_dir = os.path.join(project_root, f"tests/data/Datasets_{args.mode.capitalize()}/Dataset_AI/Raw")
    groundtruth_dir = os.path.join(project_root, f"tests/data/Datasets_{args.mode.capitalize()}/Dataset_AI/ground_truth")
    output_dir = os.path.join(project_root, f"tests/results/{args.mode.capitalize()}/whole")
    
    evaluator = Evaluator(
        configs_dir=configs_dir,
        input_dir=input_dir,
        groundtruth_dir=groundtruth_dir,
        output_dir=output_dir,
        mode=args.mode
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main() 