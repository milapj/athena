from typing import List, Optional, Tuple


def accuracy(predicted_index: Optional[int], ground_truth_index: Optional[int]) -> float:
    """Binary accuracy for multiple-choice questions."""
    if predicted_index is None or ground_truth_index is None:
        return 0.0
    return 1.0 if predicted_index == ground_truth_index else 0.0


def string_accuracy(predicted: str, ground_truth: str) -> float:
    """Fuzzy string match: 1.0 if ground truth appears in predicted (case-insensitive)."""
    return 1.0 if ground_truth.strip().lower() in predicted.strip().lower() else 0.0


def span_f1(
    predicted_spans: List[Tuple[int, int]],
    ground_truth_spans: List[Tuple[int, int]],
    total_length: int,
) -> float:
    """Token-level F1 between predicted and ground truth spans."""
    if total_length == 0:
        return 0.0

    pred_set = set()
    for start, end in predicted_spans:
        pred_set.update(range(start, end))

    gt_set = set()
    for start, end in ground_truth_spans:
        gt_set.update(range(start, end))

    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def hallucinations_per_100_tokens(
    hallucinated_token_count: int,
    total_tokens: int,
) -> float:
    """Normalized hallucination rate."""
    if total_tokens == 0:
        return 0.0
    return (hallucinated_token_count / total_tokens) * 100


def precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    """Proportion of retrieved documents that are relevant."""
    if not retrieved_doc_ids:
        return 0.0
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    """Proportion of relevant documents that were retrieved."""
    if not relevant_doc_ids:
        return 1.0  # nothing to recall
    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def retrieval_f1(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    """F1 score combining retrieval precision and recall."""
    p = precision_at_k(retrieved_doc_ids, relevant_doc_ids)
    r = recall_at_k(retrieved_doc_ids, relevant_doc_ids)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def parse_choice_letter(text: str, num_choices: int) -> Optional[int]:
    """Extract a choice index from model output like 'A', 'B)', '(C)', etc."""
    text = text.strip().upper()
    letters = "ABCDEFGHIJ"[:num_choices]

    # Try direct single letter match
    if len(text) == 1 and text in letters:
        return letters.index(text)

    # Try patterns like "A)", "(A)", "A.", "Answer: A"
    for i, letter in enumerate(letters):
        patterns = [
            f"{letter})",
            f"({letter})",
            f"{letter}.",
            f"answer: {letter}",
            f"answer is {letter}",
            f"answer is ({letter})",
        ]
        for pattern in patterns:
            if pattern in text.upper():
                return i

    # Try first character
    if text and text[0] in letters:
        return letters.index(text[0])

    return None
