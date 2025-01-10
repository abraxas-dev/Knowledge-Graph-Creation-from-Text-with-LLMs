from sklearn.metrics import precision_score, recall_score, f1_score
from rouge import Rouge
import re

# Normalize triples (basic preprocessing for comparison)
def normalize_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

def normalize_triple(triple):
    return tuple(map(normalize_text, triple))

# Exact match comparison
def exact_match(generated, ground_truth):
    generated = set(map(normalize_triple, generated))
    ground_truth = set(map(normalize_triple, ground_truth))
    tp = len(generated & ground_truth)
    fp = len(generated - ground_truth)
    fn = len(ground_truth - generated)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"Precision": precision, "Recall": recall, "F1": f1}

# Relaxed comparison (substring or partial match)
def relaxed_match(generated, ground_truth):
    def relaxed_compare(el1, el2):
        return el1 in el2 or el2 in el1
    
    def triple_relaxed_equal(tr1, tr2):
        return all(relaxed_compare(e1, e2) for e1, e2 in zip(tr1, tr2))

    tp = sum(
        1 for g in generated
        for gt in ground_truth if triple_relaxed_equal(g, gt)
    )
    fp = len(generated) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"Precision": precision, "Recall": recall, "F1": f1}

# ROUGE metrics for triples
def rouge_evaluation(generated, ground_truth):
    rouge = Rouge()
    gen_text = " ".join([" ".join(triple) for triple in generated])
    gt_text = " ".join([" ".join(triple) for triple in ground_truth])
    scores = rouge.get_scores(gen_text, gt_text, avg=True)
    return {
        "ROUGE-1": scores["rouge-1"]["f"],
        "ROUGE-2": scores["rouge-2"]["f"],
        "ROUGE-L": scores["rouge-l"]["f"],
    }

# Combined metrics function
def evaluate_triples(generated, ground_truth):
    normalized_gen = list(map(normalize_triple, generated))
    normalized_gt = list(map(normalize_triple, ground_truth))

    exact = exact_match(normalized_gen, normalized_gt)
    relaxed = relaxed_match(normalized_gen, normalized_gt)
    rouge = rouge_evaluation(normalized_gen, normalized_gt)

    return {
        "Exact Match Metrics": exact,
        "Relaxed Match Metrics": relaxed,
        "ROUGE Metrics": rouge,
    }

# Example Triples
generated_triples = [
    ("Barack Obama", "was born in", "Hawaii"),
    ("Python", "is a", "programming language"),
]

ground_truth_triples = [
    ("Barack Obama", "born in", "Hawaii"),
    ("Python", "is a", "language"),
]

# Evaluate
results = evaluate_triples(generated_triples, ground_truth_triples)

# Print Results
for metric, values in results.items():
    print(f"{metric}:")
    for key, value in values.items():
        print(f"  {key}: {value:.4f}")
