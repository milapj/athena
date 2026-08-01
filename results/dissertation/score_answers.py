"""Recompute token-level precision, recall and F1 for every reported configuration.

Reads the archived raw outputs, restricts to the 1,701-query complete-case set,
and writes results/dissertation/answer_metrics.csv.

    python results/dissertation/score_answers.py
"""
import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark.evaluation.answer_metrics import token_scores

ROOT = os.path.join(os.path.dirname(__file__), "..")
SOURCES = {
    "A1": ("mh-a1/raw_results.jsonl", open),
    "A2": ("mh-a2/raw_results.jsonl", open),
    "A3": ("mh-a3/raw_results.jsonl", open),
    "A4": ("mh-a4/raw_results.jsonl", open),
    "B1": ("mh-b/by_config/B1.jsonl.gz", gzip.open),
    "C1": ("mh-b/by_config/C1.jsonl.gz", gzip.open),
    "E1": ("mh-b/by_config/E1.jsonl.gz", gzip.open),
}


def read(rel, opener):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        alt = path.replace(".jsonl.gz", ".jsonl")
        path = alt if os.path.exists(alt) else path
        opener = open if path.endswith(".jsonl") else opener
    with opener(path, "rt") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def main():
    retained = {r["sample_id"] for r in read(*SOURCES["B1"])}
    out = []
    for name, (rel, opener) in SOURCES.items():
        p = r = f = n = 0.0, 0.0, 0.0, 0
        ps, rs, fs, em, count = [], [], [], [], 0
        for row in read(rel, opener):
            if row["sample_id"] not in retained:
                continue
            count += 1
            em.append(float(row.get("accurate") or 0.0))
            s = token_scores(row.get("predicted_answer"), row.get("ground_truth"))
            if s:
                ps.append(s[0]); rs.append(s[1]); fs.append(s[2])
        out.append({
            "config": name,
            "n": count,
            "exact_match": round(sum(em) / len(em), 4),
            "precision": round(sum(ps) / len(ps), 4),
            "recall": round(sum(rs) / len(rs), 4),
            "f1": round(sum(fs) / len(fs), 4),
            "scored": len(ps),
        })
    dest = os.path.join(os.path.dirname(__file__), "answer_metrics.csv")
    with open(dest, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    for row in out:
        print("%-3s n=%d  EM %.4f  P %.4f  R %.4f  F1 %.4f  (scored %d)"
              % (row["config"], row["n"], row["exact_match"], row["precision"],
                 row["recall"], row["f1"], row["scored"]))
    print("wrote", dest)


if __name__ == "__main__":
    main()
