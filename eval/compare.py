"""Print a markdown before/after table from eval/results/*.json files.

RAGAS metrics are compared on the questions that were scored in *every* run
(a judge failure shows up as n/a and would otherwise bias the averages), so
the numbers in one column are directly comparable across rows.

Usage:
    uv run python eval/compare.py eval/results/chunk500_k4.json eval/results/chunk1000_k4.json
"""

import json
import os
import sys

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load(path):
    with open(path) as f:
        data = json.load(f)
    data["name"] = os.path.splitext(os.path.basename(path))[0]
    return data


def paired_means(runs, metric):
    """Mean of `metric` per run, over the questions scored in every run."""
    questions = None
    for run in runs:
        scored = {s["question"] for s in run["samples"] if s.get(metric) is not None}
        questions = scored if questions is None else questions & scored
    means = []
    for run in runs:
        values = [s[metric] for s in run["samples"] if s["question"] in questions]
        means.append(sum(values) / len(values) if values else None)
    return means, len(questions)


def fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def main(paths):
    runs = [load(p) for p in paths]
    full_runs = [r for r in runs if not r["config"]["retrieval_only"]]

    print("| run | chunk | k | page hit | source hit | context chars |")
    print("|---|---|---|---|---|---|")
    for r in runs:
        c, s = r["config"], r["summary"]
        print(
            f"| {r['name']} | {c['chunk_size']}/{c['chunk_overlap']} | {c['k']} "
            f"| {fmt(s['page_hit'])} | {fmt(s['source_hit'])} | {s['context_chars']:.0f} |"
        )

    if len(full_runs) < 1:
        return
    shared = [
        m for m in RAGAS_METRICS if all(m in r["config"]["metrics"] for r in full_runs)
    ]
    paired = {m: paired_means(full_runs, m) for m in shared}
    print()
    print("| run | " + " | ".join(shared) + " |")
    print("|---|" + "---|" * len(shared))
    for i, r in enumerate(full_runs):
        cells = [fmt(paired[m][0][i]) for m in shared]
        print(f"| {r['name']} | " + " | ".join(cells) + " |")
    n_cells = ", ".join(f"{m}: {paired[m][1]}" for m in shared)
    print(f"\nQuestions scored in every run: {n_cells}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
