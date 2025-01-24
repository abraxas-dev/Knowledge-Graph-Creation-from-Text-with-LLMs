import re
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer

class TextTripleEvaluator:
    def __init__(self, gen_file, gt_file):
        """
        Initialize the TextTripleEvaluator with generated triples and ground truth triples.
        :param gen_file: Path to the generated triples text file.
        :param gt_file: Path to the ground truth triples text file.
        """
        self.generated_data = self.load_triples_from_text(gen_file)
        self.ground_truth_data = self.load_triples_from_text(gt_file)
        self.results = {}

    @staticmethod
    def load_triples_from_text(filename):
        """Load triples from a text file."""
        with open(filename, "r") as file:
            triples = [eval(line.strip()) for line in file]
        return triples

    @staticmethod
    def normalize_text(text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def strict_match(self):
        """
        Perform strict matching of triples and calculate metrics.
        """
        true_positives = len(set(self.generated_data) & set(self.ground_truth_data))
        false_positives = len(set(self.generated_data) - set(self.ground_truth_data))
        false_negatives = len(set(self.ground_truth_data) - set(self.generated_data))

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

    def relaxed_match(self):
        """
        Perform relaxed matching by checking if the ground truth triple is partially included in the generated triple.
        """
        generated_normalized = [
            (self.normalize_text(s), self.normalize_text(p), self.normalize_text(o)) for s, p, o in self.generated_data
        ]
        ground_truth_normalized = [
            (self.normalize_text(s), self.normalize_text(p), self.normalize_text(o)) for s, p, o in self.ground_truth_data
        ]

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for gt in ground_truth_normalized:
            match_found = False
            for gen in generated_normalized:
                if all(gt_comp in gen for gt_comp in gt if gt_comp):  # Check if all non-empty ground truth components are in generated triple
                    true_positives += 1
                    match_found = True
                    break
            if not match_found:
                false_negatives += 1

        for gen in generated_normalized:
            match_found = False
            for gt in ground_truth_normalized:
                if all(gt_comp in gen for gt_comp in gt if gt_comp):
                    match_found = True
                    break
            if not match_found:
                false_positives += 1

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

    def rouge_match(self):
        """
        Perform ROUGE-based evaluation for triples using ROUGE-1, ROUGE-2, and ROUGE-L.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0

        for gen in self.generated_data:
            best_rouge1 = 0
            best_rouge2 = 0
            best_rougeL = 0
            for gt in self.ground_truth_data:
                scores = scorer.score(" ".join(gen), " ".join(gt))
                best_rouge1 = max(best_rouge1, scores["rouge1"].fmeasure)
                best_rouge2 = max(best_rouge2, scores["rouge2"].fmeasure)
                best_rougeL = max(best_rougeL, scores["rougeL"].fmeasure)
            total_rouge1 += best_rouge1
            total_rouge2 += best_rouge2
            total_rougeL += best_rougeL

        return {
            "Average ROUGE-1": total_rouge1 / len(self.generated_data) if len(self.generated_data) > 0 else 0,
            "Average ROUGE-2": total_rouge2 / len(self.generated_data) if len(self.generated_data) > 0 else 0,
            "Average ROUGE-L": total_rougeL / len(self.generated_data) if len(self.generated_data) > 0 else 0,
        }

    def evaluate(self):
        """
        Evaluate triples using strict, relaxed, and ROUGE matching.
        """
        self.results['Strict Match'] = self.strict_match()
        self.results['Relaxed Match'] = self.relaxed_match()
        self.results['ROUGE Match'] = self.rouge_match()
        return self.results

# Main Execution Example
if __name__ == "__main__":
    # Input text files
    gen_file = "generated_triples.txt"
    gt_file = "groundtruth_triples.txt"

    # Initialize evaluator and evaluate
    evaluator = TextTripleEvaluator(gen_file, gt_file)
    results = evaluator.evaluate()

    # Print results to console
    print("Evaluation Results:")
    for category, metrics in results.items():
        print(f"{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
