import re
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score


class TripleEvaluator:
    def __init__(self, gen_file, gt_file):
        """
        Initialize the TripleEvaluator with generated triples and ground truth triples.
        :param gen_file: Path to the generated triples YAML file.
        :param gt_file: Path to the ground truth triples YAML file.
        """
        self.generated_data = self.load_triples_from_yaml(gen_file)
        self.ground_truth_data = self.load_triples_from_yaml(gt_file)
        self.results = {}

    @staticmethod
    def load_triples_from_yaml(filename):
        """Load triples and expected IDs from a YAML file."""
        with open(filename, "r") as file:
            data = yaml.safe_load(file)['triples']
        return data

    @staticmethod
    def normalize_text(text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    @staticmethod
    def normalize_triple(triple):
        """Normalize each element of the triple."""
        return {
            'subject': TripleEvaluator.normalize_text(triple['subject']),
            'predicate': TripleEvaluator.normalize_text(triple['predicate']),
            'object': TripleEvaluator.normalize_text(triple['object']),
        }

    def evaluate_triples(self):
        """
        Compare textual triples and calculate Precision, Recall, and F1-Score.
        """
        # Normalize textual triples
        generated_triples = [self.normalize_triple(item['triple']) for item in self.generated_data]
        ground_truth_triples = [self.normalize_triple(item['triple']) for item in self.ground_truth_data]

        # Generate true and predicted values for textual comparison
        y_true = [1 if gt in generated_triples else 0 for gt in ground_truth_triples]
        y_pred = [1 if gen in ground_truth_triples else 0 for gen in generated_triples]

        # Calculate metrics
        return {
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

    def evaluate_ids(self, component):
        """
        Compare IDs for a specific component (subject, predicate, or object) and calculate metrics.
        :param component: The key to compare (e.g., 'subject', 'predicate', 'object').
        """
        generated_ids = [item['expected_id'][component] for item in self.generated_data]
        ground_truth_ids = [item['expected_id'][component] for item in self.ground_truth_data]

        # Generate true and predicted values for ID comparison
        y_true = [1 if gt != "not_found" else 0 for gt in ground_truth_ids]
        y_pred = [1 if gen == gt and gt != "not_found" else 0 for gen, gt in zip(generated_ids, ground_truth_ids)]

        # Calculate metrics
        return {
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

    def evaluate(self):
        """Evaluate metrics for Triples, Entities, Properties, and Objects."""
        # Evaluate textual triples
        self.results['Triples'] = self.evaluate_triples()

        # Evaluate IDs for entities (subjects), properties (predicates), and objects
        self.results['Entities'] = self.evaluate_ids('subject')
        self.results['Properties'] = self.evaluate_ids('predicate')
        self.results['Objects'] = self.evaluate_ids('object')

        return self.results

    def write_results_to_yaml(self, output_file):
        """Write evaluation results to a YAML file."""
        with open(output_file, "w") as file:
            yaml.dump(self.results, file, default_flow_style=False, allow_unicode=True)


# Main Execution Example
if __name__ == "__main__":
    # Input YAML files
    gen_file = "generated_triples.yaml"
    gt_file = "groundtruth_triples.yaml"
    output_file = "evaluation_results.yaml"

    # Initialize evaluator and evaluate
    evaluator = TripleEvaluator(gen_file, gt_file)
    results = evaluator.evaluate()

    # Write results to a YAML file
    evaluator.write_results_to_yaml(output_file)

    # Print results to console
    print("Evaluation Results:")
    for category, metrics in results.items():
        print(f"{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
