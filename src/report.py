"""Render experiment summaries for the terminal and project reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

CATEGORY_NAMES = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
}

SummaryRows = List[Tuple[str, dict]]


def _percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def _overall_row(name: str, metrics: dict) -> List[str]:
    retrieval = metrics["retrieval"]
    return [
        name,
        str(metrics["n"]),
        _percent(metrics["qa_accuracy"]),
        _percent(metrics["exact_match"]),
        _percent(metrics["token_f1"]),
        _percent(retrieval["evidence_hit_at_k"]),
        _percent(retrieval["evidence_recall_at_k"]),
        f"{retrieval['mrr']:.3f}",
        _percent(metrics["empty_retrieval_rate"]),
    ]


def _answer_row(name: str, metrics: dict) -> List[str]:
    return [
        name,
        str(metrics["n"]),
        _percent(metrics["qa_accuracy"]),
        f"{metrics['judge_score']:.3f}",
        _percent(metrics["exact_match"]),
        _percent(metrics["token_f1"]),
    ]


def _retrieval_row(name: str, metrics: dict) -> List[str]:
    retrieval = metrics["retrieval"]
    return [
        name,
        str(retrieval["n_with_evidence"]),
        _percent(retrieval["evidence_hit_at_k"]),
        _percent(retrieval["evidence_precision_at_k"]),
        _percent(retrieval["evidence_recall_at_k"]),
        _percent(retrieval["evidence_full_recall_at_k"]),
        f"{retrieval['mrr']:.3f}",
        f"{metrics['avg_memories_retrieved']:.2f}",
        _percent(metrics["empty_retrieval_rate"]),
    ]


def _category_rows(rows: SummaryRows) -> List[List[str]]:
    result = []
    for name, metrics in rows:
        for category, category_metrics in metrics["by_category"].items():
            retrieval = category_metrics["retrieval"]
            result.append([
                name,
                f"{category}: {CATEGORY_NAMES.get(category, 'Unknown')}",
                str(category_metrics["n"]),
                _percent(category_metrics["qa_accuracy"]),
                _percent(category_metrics["token_f1"]),
                _percent(retrieval["evidence_hit_at_k"]),
                _percent(retrieval["evidence_recall_at_k"]),
            ])
    return result


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(rows: SummaryRows, k: int) -> str:
    answer_headers = [
        "Method",
        "N",
        "Judge Acc.",
        "Mean Judge Score",
        "Exact Match",
        "Token F1",
    ]
    retrieval_headers = [
        "Method",
        "Evidence N",
        f"Evidence Hit@{k}",
        f"Evidence Precision@{k}",
        f"Evidence Recall@{k}",
        f"Full Recall@{k}",
        "MRR",
        "Avg. Memories",
        "Empty Retrieval",
    ]
    category_headers = [
        "Method",
        "Category",
        "N",
        "Judge Acc.",
        "Token F1",
        f"Evidence Hit@{k}",
        f"Evidence Recall@{k}",
    ]
    return "\n".join([
        "# Evaluation Report",
        "",
        f"Retrieval metrics use the top {k} memories. Evidence metrics are macro-averaged "
        "over questions that provide gold evidence.",
        "",
        "## Overall comparison",
        "",
        "### Answer quality",
        "",
        _markdown_table(answer_headers, [_answer_row(name, metrics) for name, metrics in rows]),
        "",
        "### Retrieval quality",
        "",
        _markdown_table(
            retrieval_headers,
            [_retrieval_row(name, metrics) for name, metrics in rows],
        ),
        "",
        "## LOCOMO category breakdown",
        "",
        _markdown_table(category_headers, _category_rows(rows)),
        "",
        "Categories: 1 = Multi-hop, 2 = Temporal, 3 = Open-domain, "
        "4 = Single-hop.",
        "",
    ])


def render_console(rows: SummaryRows, k: int) -> str:
    headers = [
        "combo",
        "n",
        "judge",
        "EM",
        "F1",
        f"hit@{k}",
        f"recall@{k}",
        "MRR",
        "empty",
    ]
    table_rows = [_overall_row(name, metrics) for name, metrics in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def line(values):
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    return "\n".join([line(headers), line(["-" * width for width in widths]), *map(line, table_rows)])


def write_reports(rows: SummaryRows, output_dir: Path, k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrieval_k": k,
        "runs": [{"combo": name, **metrics} for name, metrics in rows],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with open(output_dir / "report.md", "w", encoding="utf-8") as file:
        file.write(render_markdown(rows, k))
