import re
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge import Rouge

class TripleEvaluator:
    def __init__(self, generated_triples, ground_truth_triples):
        # Initialize the triples and normalize them
        self.generated_triples = list(map(self.normalize_triple, generated_triples))
        self.ground_truth_triples = list(map(self.normalize_triple, ground_truth_triples))
        self.results = {}

    @staticmethod
    def normalize_text(text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    @staticmethod
    def normalize_triple(triple):
        """Normalize each element of the triple."""
        return tuple(map(TripleEvaluator.normalize_text, triple))

    @staticmethod
    def relaxed_compare(el1, el2):
        """Check if one element is a substring of the other."""
        return el1 in el2 or el2 in el1

    @staticmethod
    def relaxed_triple_match(triple1, triple2):
        """Check if two triples match using relaxed criteria."""
        return all(TripleEvaluator.relaxed_compare(e1, e2) for e1, e2 in zip(triple1, triple2))

    def exact_y_true(self):
        """Check if generated triples exactly match ground truth triples."""
        return [1 if triple in self.ground_truth_triples else 0 for triple in self.generated_triples]

    def relaxed_y_true(self):
        """Check if generated triples match ground truth triples under relaxed criteria."""
        return [
            1 if any(self.relaxed_triple_match(g, gt) for gt in self.ground_truth_triples) else 0
            for g in self.generated_triples
        ]

    @staticmethod
    def evaluate_metrics(y_true, y_pred):
        """Compute precision, recall, and F1 score."""
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"Precision": precision, "Recall": recall, "F1": f1}

    def rouge_evaluation(self):
        """Compute ROUGE metrics."""
        rouge = Rouge()
        gen_text = " ".join([" ".join(triple) for triple in self.generated_triples])
        gt_text = " ".join([" ".join(triple) for triple in self.ground_truth_triples])
        scores = rouge.get_scores(gen_text, gt_text, avg=True)
        return {
            "ROUGE-1": scores["rouge-1"]["f"],
            "ROUGE-2": scores["rouge-2"]["f"],
            "ROUGE-L": scores["rouge-l"]["f"],
        }

    def evaluate(self):
        """Evaluate exact, relaxed, and ROUGE metrics."""
        y_pred = [1] * len(self.generated_triples)

        # Exact matching
        y_true_exact = self.exact_y_true()
        self.results["Exact Matching Metrics"] = self.evaluate_metrics(y_true_exact, y_pred)

        # Relaxed matching
        y_true_relaxed = self.relaxed_y_true()
        self.results["Relaxed Matching Metrics"] = self.evaluate_metrics(y_true_relaxed, y_pred)

        # ROUGE evaluation
        self.results["ROUGE Metrics"] = self.rouge_evaluation()

        return self.results

    def write_results_to_file(self, filename):
        """Write evaluation results to an external file."""
        with open(filename, "w") as f:
            for match_type, metrics in self.results.items():
                f.write(f"{match_type}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")

# Main execution
if __name__ == "__main__":
    # Example triples
    generated_triples = [
        ("Barack Obama", "was born in", "Hawaii"),
        ("Python", "is a", "programming language"),
    ]

    ground_truth_triples = [
        ("Barack Obama", "born in", "Hawaii"),
        ("Python", "is a", "language"),
    ]

    # Initialize the evaluator
    evaluator = TripleEvaluator(generated_triples, ground_truth_triples)

    # Perform evaluation
    results = evaluator.evaluate()

    # Print results
    for match_type, metrics in results.items():
        print(f"{match_type}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Write results to a file
    evaluator.write_results_to_file("evaluation_results.txt")
