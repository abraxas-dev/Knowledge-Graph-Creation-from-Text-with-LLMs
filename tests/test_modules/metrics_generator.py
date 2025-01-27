import re
from ast import literal_eval
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score


class TextTripleEvaluator:
    def __init__(self, valid_file, gt_file):
        """
        Initialize the TextTripleEvaluator with valid triples and ground truth triples.
        Count lines from the invalid triples file.
        :param valid_file: Path to the valid triples text file.
        :param invalid_file: Path to the invalid triples text file.
        :param gt_file: Path to the ground truth triples text file.
        """
        self.valid_data = self.load_triples_from_text(valid_file)
        #self.invalid_count = self.count_lines(invalid_file)
        self.ground_truth_data = self.load_triples_from_text(gt_file)
        self.results = {}

    def load_triples_from_text(self, filename):
        """Load triples from a text file."""
        triples = []
        try:
            with open(filename, "r", encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # Remove trailing semicolon if present
                        if line.endswith(';'):
                            line = line[:-1]
                        
                        # Remove outer parentheses if present
                        line = line.strip('()')
                        
                        # Split by comma and handle quotes
                        parts = []
                        current = []
                        in_quotes = False
                        
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                parts.append(''.join(current).strip())
                                current = []
                                continue
                            current.append(char)
                        
                        # Add the last part
                        if current:
                            parts.append(''.join(current).strip())
                        
                        # Clean up parts (remove quotes and extra whitespace)
                        parts = [p.strip().strip('"') for p in parts]
                        
                        if len(parts) == 3:
                            triples.append(tuple(parts))
                        else:
                            print(f"Warning: Skipping malformed triple in {filename}: {line}")
                            
                    except Exception as e:
                        print(f"Warning: Error parsing line in {filename}: {line} - {str(e)}")
                        continue
                        
            return triples
        except Exception as e:
            raise ValueError(f"Error reading {filename}: {e}")

    @staticmethod
    def count_lines(filename):
        """Count the number of lines in a file."""
        try:
            with open(filename, "r") as file:
                return sum(1 for _ in file if _.strip())
        except FileNotFoundError as e:
            raise ValueError(f"Error reading {filename}: {e}")

    @staticmethod
    def normalize_text(text):
        """Normalize text by converting to lowercase and removing special characters."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def calculate_valid_percentage(self):
        """
        Calculate the percentage of valid triples compared to invalid triples.
        """
        valid_count = len(self.valid_data)
        invalid_count = self.invalid_count

        if invalid_count == 0:
            return 100.0 if valid_count > 0 else 0.0  # Handle edge case when no invalid triples exist.

        percentage = (valid_count / (valid_count + invalid_count)) * 100
        return {
            "Valid Count": valid_count,
            "Invalid Count": invalid_count,
            "Valid Percentage": percentage,
        }

    def strict_match(self):
        """
        Perform strict matching of triples and calculate metrics using valid triples.
        """
        generated_set = set(map(tuple, self.valid_data))
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
        Perform relaxed matching by checking if the ground truth triple is partially included in the valid triples.
        """
        generated_normalized = [
            (self.normalize_text(s), self.normalize_text(p), self.normalize_text(o)) for s, p, o in self.valid_data
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
        Perform ROUGE-based evaluation for triples using valid triples.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        total_rouge1 = 0
        total_rouge2 = 0
        total_rougeL = 0

        for gen in self.valid_data:
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
            "Average ROUGE-1": total_rouge1 / len(self.valid_data) if self.valid_data else 0,
            "Average ROUGE-2": total_rouge2 / len(self.valid_data) if self.valid_data else 0,
            "Average ROUGE-L": total_rougeL / len(self.valid_data) if self.valid_data else 0,
        }

    def bert_score_match(self):
        """
        Perform BERTScore evaluation for valid triples.
        """
        generated_texts = [" ".join(triple) for triple in self.valid_data]
        ground_truth_texts = [" ".join(triple) for triple in self.ground_truth_data]

        precision, recall, f1 = bert_score(
            generated_texts, ground_truth_texts, lang="en", verbose=False
        )

        return {
            "Average BERTScore Precision": precision.mean().item(),
            "Average BERTScore Recall": recall.mean().item(),
            "Average BERTScore F1": f1.mean().item(),
        }

    def evaluate(self):
        """
        Evaluate triples using strict, relaxed, ROUGE, and BERTScore matching.
        """
        #self.results["Valid Percentage"] = self.calculate_valid_percentage()
        self.results["Strict Match"] = self.strict_match()
        self.results["Relaxed Match"] = self.relaxed_match()
        self.results["ROUGE Match"] = self.rouge_match()
        self.results["BERTScore Match"] = self.bert_score_match()
        return self.results


# Main Execution
if __name__ == "__main__":
    # Hardcoded file paths
    valid_file = "valid_triples.txt"
    invalid_file = "invalid_triples.txt"
    gt_file = "groundtruth_triples.txt"

    evaluator = TextTripleEvaluator(valid_file, invalid_file, gt_file)
    results = evaluator.evaluate()

    print("Evaluation Results:")
    for category, metrics in results.items():
        print(f"{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
