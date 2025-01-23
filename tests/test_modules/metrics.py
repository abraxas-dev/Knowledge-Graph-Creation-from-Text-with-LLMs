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

    def evaluate_triples(self):
        """
        Compare triples based on IDs (subject_id, predicate_id, object_id) and calculate Precision, Recall, and F1-Score.
        """
        # Extract IDs from generated and ground truth triples
        generated_ids = [
            (item['expected_id']['subject'], item['expected_id']['predicate'], item['expected_id']['object'])
            for item in self.generated_data
        ]
        ground_truth_ids = [
            (item['expected_id']['subject'], item['expected_id']['predicate'], item['expected_id']['object'])
            for item in self.ground_truth_data
        ]

        # Initialize counters for true positives, false positives, and false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Calculate True Positives and False Negatives
        for gt_id in ground_truth_ids:
            if gt_id in generated_ids:
                true_positives += 1
            else:
                false_negatives += 1

        # Calculate False Positives
        for gen_id in generated_ids:
            if gen_id not in ground_truth_ids:
                false_positives += 1

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True Positives": true_positives,
            "False Positives": false_positives,
            "False Negatives": false_negatives,
        }

    def evaluate_ids(self, component):
        """
        Compare IDs for a specific component (subject, predicate, or object) and calculate metrics.
        :param component: The key to compare (e.g., 'subject', 'predicate', 'object').
        """
        generated_ids = [item['expected_id'][component] for item in self.generated_data]
        ground_truth_ids = [item['expected_id'][component] for item in self.ground_truth_data]

        # Initialize counters for true positives, false positives, and false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Calculate True Positives and False Negatives
        for gt_id in ground_truth_ids:
            if gt_id in generated_ids:
                true_positives += 1
            else:
                false_negatives += 1

        # Calculate False Positives
        for gen_id in generated_ids:
            if gen_id not in ground_truth_ids:
                false_positives += 1

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True Positives": true_positives,
            "False Positives": false_positives,
            "False Negatives": false_negatives,
        }

    def evaluate(self):
        """Evaluate metrics for Triples, Entities, Properties, and Objects."""
        # Evaluate triples based on IDs
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
