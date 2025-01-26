import re
from ast import literal_eval
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class TextTripleEvaluator:
    def __init__(self, generated_file, gt_file):
        """
        Initialize the TextTripleEvaluator with generated triples and ground truth triples.
        :param generated_file: Path to the generated triples text file.
        :param gt_file: Path to the ground truth triples text file.
        """
        self.generated_data = self.load_triples_from_text(generated_file)
        self.ground_truth_data = self.load_triples_from_text(gt_file)
        self.results = {}

    @staticmethod
    def load_triples_from_text(filename):
        """Load triples from a text file safely using literal_eval."""
        try:
            with open(filename, "r") as file:
                triples = [literal_eval(line.strip()) for line in file if line.strip()]
            return triples
        except (FileNotFoundError, SyntaxError, ValueError) as e:
            raise ValueError(f"Error reading {filename}: {e}")

    @staticmethod
    def normalize_text(text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def strict_match(self):
        """
        Perform strict matching of triples and calculate metrics using generated triples.
        """
        generated_set = set(map(tuple, self.generated_data))
        ground_truth_set = set(map(tuple, self.ground_truth_data))

        true_positives = len(generated_set & ground_truth_set)
        false_positives = len(generated_set - ground_truth_set)
        false_negatives = len(ground_truth_set - generated_set)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

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
        Perform relaxed matching by checking if the ground truth triple is partially included in the generated triples.
        """
        generated_normalized = [
            (self.normalize_text(s), self.normalize_text(p), self.normalize_text(o)) for s, p, o in self.generated_data
        ]
        ground_truth_normalized = [
            (self.normalize_text(s), self.normalize_text(p), self.normalize_text(o)) for s, p, o in self.ground_truth_data
        ]

        true_positives = sum(
            any(all(gt_comp in gen for gt_comp in gt if gt_comp) for gen in generated_normalized)
            for gt in ground_truth_normalized
        )
        false_negatives = len(ground_truth_normalized) - true_positives
        false_positives = len(generated_normalized) - true_positives

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

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
        Perform ROUGE-based evaluation for triples using generated triples.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0

        for gen in self.generated_data:
            best_rouge1 = best_rouge2 = best_rougeL = 0
            for gt in self.ground_truth_data:
                scores = scorer.score(" ".join(gen), " ".join(gt))
                best_rouge1 = max(best_rouge1, scores["rouge1"].fmeasure)
                best_rouge2 = max(best_rouge2, scores["rouge2"].fmeasure)
                best_rougeL = max(best_rougeL, scores["rougeL"].fmeasure)
            total_rouge1 += best_rouge1
            total_rouge2 += best_rouge2
            total_rougeL += best_rougeL

        return {
            "Average ROUGE-1": total_rouge1 / len(self.generated_data) if self.generated_data else 0,
            "Average ROUGE-2": total_rouge2 / len(self.generated_data) if self.generated_data else 0,
            "Average ROUGE-L": total_rougeL / len(self.generated_data) if self.generated_data else 0,
        }

    

    def evaluate(self):
        """
        Evaluate triples using strict, relaxed, ROUGE, and BERTScore matching.
        """
        self.results["Strict Match"] = self.strict_match()
        self.results["Relaxed Match"] = self.relaxed_match()
        self.results["ROUGE Match"] = self.rouge_match()
    
        return self.results


# Main Execution
if __name__ == "__main__":
    # Hardcoded file paths
    generated_file = "generatedlebron.txt"
    gt_file = "groundtruthLebron.txt"

    evaluator = TextTripleEvaluator(generated_file, gt_file)
    results = evaluator.evaluate()

    print("Evaluation Results:")
    for category, metrics in results.items():
        print(f"{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")


