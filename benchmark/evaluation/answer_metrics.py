"""Token-level answer metrics for short-answer question answering.

The dissertation reports Exact Match and token-level Recall. Exact Match is the
containment check in metrics.string_accuracy: the normalized gold answer must
appear in the normalized model response. The token-level measures below are
computed over the same normalization so the two are directly comparable.

Normalization follows the convention established for SQuAD (Rajpurkar et al.,
2016): lowercase, strip punctuation, drop the articles a/an/the, and split on
whitespace. Overlap is counted as a multiset intersection, so a token repeated
in the gold answer must be repeated in the response to count twice.

Recall is the share of gold tokens recovered. Precision is the share of response
tokens that are gold tokens, which penalizes padding an answer with unsupported
text. F1 is their harmonic mean.
"""
import collections
import re
import string

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> list:
    """Lowercase, strip punctuation, drop articles, split on whitespace."""
    s = (text or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return s.split()


def token_scores(predicted: str, ground_truth: str):
    """Return (precision, recall, f1) over normalized tokens.

    Returns None when the gold answer normalizes to nothing, so such items can
    be excluded rather than silently scored as zero.
    """
    pred = normalize(predicted)
    gold = normalize(ground_truth)
    if not gold:
        return None
    if not pred:
        return (0.0, 0.0, 0.0)

    overlap = sum((collections.Counter(pred) & collections.Counter(gold)).values())
    if overlap == 0:
        return (0.0, 0.0, 0.0)

    precision = overlap / len(pred)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)
