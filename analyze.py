"""Generate paper-ready figures and tables from a results run folder.

Reads <run_dir>/summary.json (written by run.py) and writes figures and tables
into <run_dir>/figures/. Run it after an experiment to produce the artifacts a
paper needs, without re-running any model.

Usage:
    python analyze.py                # newest folder under results/
    python analyze.py results/NAME   # a specific run folder

Outputs (in <run_dir>/figures/):
    matrix_accuracy.{pdf,png}    extractor x retriever heatmap of QA accuracy
    by_category.{pdf,png}        per-category accuracy, best retriever per extractor
    retrieval_vs_answer.{pdf,png}  evidence recall@k vs QA accuracy, one point per combo
    table_main.{md,tex}          the extractor x retriever matrix
    table_by_category.md         per-category accuracy per combo
    ablations.md                 regex v1 vs v2, word2vec uniform vs tfidf pooling
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTRACTOR_ORDER = ["append_all", "regex", "regex_v2", "ner"]
RETRIEVER_ORDER = ["tfidf", "bm25", "word2vec", "word2vec_tfidf", "sentence_emb"]
CATEGORY_NAMES = {"1": "Multi-hop", "2": "Temporal", "3": "Open-domain", "4": "Single-hop"}
BASELINE = "no_memory+no_retrieval"

# Okabe-Ito palette: colorblind-safe categorical, assigned in fixed order.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_runs(run_dir: Path):
    summary = run_dir / "summary.json"
    if not summary.exists():
        sys.exit(f"no summary.json in {run_dir}; run.py writes it after a run")
    data = json.loads(summary.read_text())
    runs = {r["combo"]: r for r in data["runs"]}
    return runs, data.get("retrieval_k", 5)


def _present(runs, order, side):
    """Keep the canonical order, but only entries that actually appear in a combo."""
    seen = {combo.split("+", 1)[0 if side == "e" else 1] for combo in runs}
    result = [x for x in order if x in seen]
    result += sorted(x for x in seen if x not in order)  # any unexpected names, appended
    return result


def qa(run):
    return run["qa_accuracy"]


def recall(run):
    return run["retrieval"]["evidence_recall_at_k"]


def best_retriever(runs, extractor, retrievers):
    """Retriever giving this extractor its highest QA accuracy."""
    scored = [(qa(runs[f"{extractor}+{r}"]), r) for r in retrievers if f"{extractor}+{r}" in runs]
    return max(scored)[1] if scored else None


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{100 * x:.1f}"


def table_main(runs, extractors, retrievers) -> str:
    head = "| Extractor \\ Retriever | " + " | ".join(retrievers) + " |"
    sep = "| " + " | ".join(["---"] * (len(retrievers) + 1)) + " |"
    lines = [head, sep]
    for e in extractors:
        cells = [_pct(qa(runs[f"{e}+{r}"])) if f"{e}+{r}" in runs else "--" for r in retrievers]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    lines.append("")
    if BASELINE in runs:
        lines.append(f"Baseline (no memory): {_pct(qa(runs[BASELINE]))}")
    lines.append("")
    lines.append("Cells are QA accuracy (%) judged by the LLM judge.")
    return "\n".join(lines)


def table_main_latex(runs, extractors, retrievers) -> str:
    cols = "l" + "r" * len(retrievers)
    header = " & ".join(["Extractor"] + [r.replace("_", r"\_") for r in retrievers]) + r" \\"
    body = []
    for e in extractors:
        cells = [_pct(qa(runs[f"{e}+{r}"])) if f"{e}+{r}" in runs else "--" for r in retrievers]
        body.append(" & ".join([e.replace("_", r"\_")] + cells) + r" \\")
    base = _pct(qa(runs[BASELINE])) if BASELINE in runs else "--"
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{QA accuracy (\%) for each extractor $\times$ retriever combination. "
        rf"No-memory baseline: {base}\%.}}",
        r"\label{tab:main}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def table_by_category(runs, extractors, retrievers) -> str:
    cats = [c for c in CATEGORY_NAMES]
    head = "| Combo | " + " | ".join(CATEGORY_NAMES[c] for c in cats) + " |"
    sep = "| " + " | ".join(["---"] * (len(cats) + 1)) + " |"
    lines = [head, sep]
    combos = [f"{e}+{r}" for e in extractors for r in retrievers if f"{e}+{r}" in runs]
    if BASELINE in runs:
        combos = [BASELINE] + combos
    for combo in combos:
        by_cat = runs[combo].get("by_category", {})
        cells = [_pct(by_cat[c]["qa_accuracy"]) if c in by_cat else "--" for c in cats]
        lines.append(f"| {combo} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per-category QA accuracy (%).")
    return "\n".join(lines)


def table_ablations(runs, extractors, retrievers) -> str:
    lines = []
    # regex v1 vs v2, per retriever.
    if "regex" in extractors and "regex_v2" in extractors:
        lines += ["## Regex rules: v1 vs v2", "",
                  "| Retriever | v1 | v2 | delta |", "| --- | --- | --- | --- |"]
        for r in retrievers:
            a, b = f"regex+{r}", f"regex_v2+{r}"
            if a in runs and b in runs:
                d = qa(runs[b]) - qa(runs[a])
                lines.append(f"| {r} | {_pct(qa(runs[a]))} | {_pct(qa(runs[b]))} | {100 * d:+.1f} |")
        lines.append("")
    # word2vec uniform vs tfidf pooling, per extractor.
    if "word2vec" in retrievers and "word2vec_tfidf" in retrievers:
        lines += ["## Word2vec pooling: uniform vs tfidf", "",
                  "| Extractor | uniform | tfidf | delta |", "| --- | --- | --- | --- |"]
        for e in extractors:
            a, b = f"{e}+word2vec", f"{e}+word2vec_tfidf"
            if a in runs and b in runs:
                d = qa(runs[b]) - qa(runs[a])
                lines.append(f"| {e} | {_pct(qa(runs[a]))} | {_pct(qa(runs[b]))} | {100 * d:+.1f} |")
        lines.append("")
    lines.append("Delta is percentage points (v2 minus v1; tfidf minus uniform).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, out_dir, name, extra=None):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)


def fig_heatmap(runs, extractors, retrievers, out_dir):
    # Magnitude -> sequential single hue, light to dark. Values labeled in each cell.
    grid = [[qa(runs[f"{e}+{r}"]) if f"{e}+{r}" in runs else float("nan")
             for r in retrievers] for e in extractors]
    fig, ax = plt.subplots(figsize=(1.2 * len(retrievers) + 1.5, 0.9 * len(extractors) + 1.5))
    im = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(retrievers)), retrievers, rotation=30, ha="right")
    ax.set_yticks(range(len(extractors)), extractors)
    ax.grid(False)
    vmax = max((v for row in grid for v in row if v == v), default=1.0)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == v:
                ax.text(j, i, _pct(v), ha="center", va="center",
                        color="white" if v > 0.6 * vmax else "#222222", fontsize=9)
    ax.set_title("QA accuracy (%) by extractor x retriever")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="QA accuracy")
    _save(fig, out_dir, "matrix_accuracy")


def fig_by_category(runs, extractors, retrievers, out_dir):
    cats = list(CATEGORY_NAMES)
    labels = [CATEGORY_NAMES[c] for c in cats]
    series = []  # (label, [acc per category])
    for e in extractors:
        r = best_retriever(runs, e, retrievers)
        if r is None:
            continue
        by_cat = runs[f"{e}+{r}"].get("by_category", {})
        series.append((f"{e} (+{r})",
                       [by_cat[c]["qa_accuracy"] if c in by_cat else float("nan") for c in cats]))
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(series)
    width = 0.8 / n
    x = list(range(len(cats)))
    for idx, (label, values) in enumerate(series):
        offset = (idx - (n - 1) / 2) * width
        ax.bar([xi + offset for xi in x], [100 * v for v in values], width,
               label=label, color=OKABE_ITO[idx % len(OKABE_ITO)])
    ax.set_xticks(x, labels)
    ax.set_ylabel("QA accuracy (%)")
    ax.set_title("QA accuracy by question category (best retriever per extractor)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, out_dir, "by_category", extra=[leg])


def fig_retrieval_vs_answer(runs, extractors, retrievers, k, out_dir):
    # One axis each; color = retriever identity, marker = extractor identity.
    fig, ax = plt.subplots(figsize=(7, 5))
    for combo, run in runs.items():
        e, r = combo.split("+", 1)
        if r == "no_retrieval":
            continue
        ci = retrievers.index(r) if r in retrievers else 0
        mi = extractors.index(e) if e in extractors else 0
        ax.scatter(100 * recall(run), 100 * qa(run), s=55,
                   color=OKABE_ITO[ci % len(OKABE_ITO)],
                   marker=MARKERS[mi % len(MARKERS)],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel(f"Evidence recall@{k} (%)")
    ax.set_ylabel("QA accuracy (%)")
    ax.set_title("Retrieval quality vs answer quality (one point per combination)")
    ax.set_axisbelow(True)
    color_handles = [plt.Line2D([], [], marker="o", linestyle="", color=OKABE_ITO[i % len(OKABE_ITO)],
                                label=r) for i, r in enumerate(retrievers)]
    marker_handles = [plt.Line2D([], [], marker=MARKERS[i % len(MARKERS)], linestyle="", color="#555555",
                                 label=e) for i, e in enumerate(extractors)]
    leg1 = ax.legend(handles=color_handles, title="Retriever", frameon=False,
                     fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=marker_handles, title="Extractor", frameon=False,
                     fontsize=8, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    _save(fig, out_dir, "retrieval_vs_answer", extra=[leg1, leg2])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def latest_run(root: Path):
    subdirs = [d for d in root.iterdir() if d.is_dir()] if root.exists() else []
    return max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else None


def generate(run_dir):
    """Write all tables and figures for one run folder into <run_dir>/figures/."""
    run_dir = Path(run_dir)
    runs, k = load_runs(run_dir)
    extractors = [e for e in _present(runs, EXTRACTOR_ORDER, "e") if e != "no_memory"]
    retrievers = [r for r in _present(runs, RETRIEVER_ORDER, "r") if r != "no_retrieval"]

    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    (out_dir / "table_main.md").write_text(table_main(runs, extractors, retrievers))
    (out_dir / "table_main.tex").write_text(table_main_latex(runs, extractors, retrievers))
    (out_dir / "table_by_category.md").write_text(table_by_category(runs, extractors, retrievers))
    (out_dir / "ablations.md").write_text(table_ablations(runs, extractors, retrievers))

    fig_heatmap(runs, extractors, retrievers, out_dir)
    fig_by_category(runs, extractors, retrievers, out_dir)
    fig_retrieval_vs_answer(runs, extractors, retrievers, k, out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="results run folder; default is the newest under results/")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(Path(__file__).parent / "results")
    if run_dir is None:
        sys.exit("no run folder found under results/; run.py first")

    out_dir = generate(run_dir)
    print(f"figures and tables written to {out_dir}")


if __name__ == "__main__":
    main()
