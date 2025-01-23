import re
from rouge import Rouge

class TripleEvaluator:
    def __init__(self, generated_triples, ground_truth_triples):
        # Initialize and normalize triples
        self.generated_triples = list(map(self.normalize_triple, generated_triples))
        self.ground_truth_triples = list(map(self.normalize_triple, ground_truth_triples))
        self.results = {}
        print("Generated Triples (Normalized):", self.generated_triples)
        print("Ground Truth Triples (Normalized):", self.ground_truth_triples)

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
        y_true = [1 if triple in self.ground_truth_triples else 0 for triple in self.generated_triples]
        print("y_true (Exact):", y_true)
        return y_true

    def relaxed_y_true(self):
        """Check if generated triples match ground truth triples under relaxed criteria."""
        y_true = [
            1 if any(self.relaxed_triple_match(g, gt) for gt in self.ground_truth_triples) else 0
            for g in self.generated_triples
        ]
        print("y_true (Relaxed):", y_true)
        return y_true

    def f1_evaluation(self, y_true, y_pred):
        """Compute precision, recall, and F1 score manually."""
        TP = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))  # True Positives
        FP = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true, y_pred))  # False Positives
        FN = len(self.ground_truth_triples) - len(self.generated_triples)  # False Negatives

        print(f"Debug: TP={TP}, FP={FP}, FN={FN}")

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"Precision": precision, "Recall": recall, "F1": f1}

    def rouge_evaluation(self):
        """Compute ROUGE metrics."""
        rouge = Rouge()
        gen_text = " ".join([" ".join(triple) for triple in self.generated_triples])
        gt_text = " ".join([" ".join(triple) for triple in self.ground_truth_triples])
        print("Generated Text for ROUGE:", gen_text)
        print("Ground Truth Text for ROUGE:", gt_text)
        scores = rouge.get_scores(gen_text, gt_text, avg=True)
        return {
            "ROUGE-1": scores["rouge-1"]["f"],
            "ROUGE-2": scores["rouge-2"]["f"],
            "ROUGE-L": scores["rouge-l"]["f"],
        }

    def evaluate(self):
        """Evaluate exact, relaxed, and ROUGE metrics."""
        y_pred = [1] * len(self.generated_triples)  # Assume all generated triples are predicted
        print("y_pred:", y_pred)

        # Exact matching
        y_true_exact = self.exact_y_true()
        self.results["Exact Matching Metrics"] = self.f1_evaluation(y_true_exact, y_pred)

        # Relaxed matching
        y_true_relaxed = self.relaxed_y_true()
        self.results["Relaxed Matching Metrics"] = self.f1_evaluation(y_true_relaxed, y_pred)

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
    ("Isaac Newton", "formulated", "laws of motion"),
    ("Isaac Newton", "discovered", "gravity"),
    ("Marie Curie", "conducted pioneering research on", "radioactivity"),
    ("Marie Curie", "won", "two Nobel Prizes"),
    ("Eiffel Tower", "located in", "Paris"),
    ("Eiffel Tower", "completed in", "1889"),
    ("Barack Obama", "was the president of", "United States"),
    ("Obama", "born in", "Hawaii"),
    ("Python", "used for", "AI"),
    ("Python", "popular in", "data science")
    ]

    ground_truth_triples = [
    ("Isaac Newton", "formulated", "laws of motion"),
    ("Isaac Newton", "discovered", "gravity"),
    ("Marie Curie", "conducted research on", "radioactivity"),
    ("Marie Curie", "won", "two Nobel Prizes"),
    ("The Eiffel Tower", "is located in", "Paris"),
    ("The Eiffel Tower", "was completed in", "1889"),
    ("Barack Obama", "served as", "44th president of the United States"),
    ("Barack Obama", "was born in", "Hawaii"),
    ("Python", "is used for", "data science"),
    ("Python", "is popular for", "artificial intelligence"),
    ("Just", "false", "negatives"),
    ("Just", "false", "negatives"),
    ("Just", "false", "negatives")
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
