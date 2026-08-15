"""Run the full LOCOMO extraction x retrieval experiment.

Two passes keep one model in memory at a time: load the reader and answer all
questions, then load the judge and score all answers. By default, the script
runs both the base and timestamp-augmented grids. Results go to
results/<name>/{base,timestamp}; analyze.generate then writes figures and
tables for each variant.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from locomo.loader import load_locomo
from src.pipeline import read_pass, judge_pass
from src.metrics import exact_match, source_dia_ids, summarize, token_f1
from src.report import render_console, write_reports

from src.extract.no_memory import NoMemory
from src.extract.append_all import AppendAll
from src.extract.regex import Regex
from src.extract.ner import NER
from src.retrieve.no_retrieval import NoRetrieval
from src.retrieve.tfidf import Tfidf
from src.retrieve.bm25 import Bm25
from src.retrieve.word2vec import Word2vec, MODEL as W2V_MODEL
from src.retrieve.sentence_emb import SentenceEmb
from src.reader import LocalLLMReader
from src.judge import LocalLLMJudge

# The two sentence encoders are separate retrievers in the experimental grid.
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
E5_MODEL = "intfloat/e5-small-v2"

# Heavy retriever models are loaded once and reused across all combos.
_shared = {}


def _w2v_vectors():
    if "w2v" not in _shared:
        import gensim.downloader
        _shared["w2v"] = gensim.downloader.load(W2V_MODEL)
    return _shared["w2v"]


def _sent_encoder(model_name):
    key = f"sent:{model_name}"
    if key not in _shared:
        from sentence_transformers import SentenceTransformer
        _shared[key] = SentenceTransformer(model_name)
    return _shared[key]


RETRIEVERS = {
    "no_retrieval": NoRetrieval,
    "tfidf": Tfidf,
    "bm25": Bm25,
    "word2vec": lambda: Word2vec(keyed_vectors=_w2v_vectors()),                        # uniform pooling
    "word2vec_tfidf": lambda: Word2vec(keyed_vectors=_w2v_vectors(), weighting="tfidf"),
    "sentence_emb": lambda: SentenceEmb(
        model_name=MINILM_MODEL,
        encoder=_sent_encoder(MINILM_MODEL),
    ),
    "sentence_emb_e5": lambda: SentenceEmb(
        model_name=E5_MODEL,
        encoder=_sent_encoder(E5_MODEL),
    ),
}
K = 5
RESULTS_ROOT = Path(__file__).parent / "results"


def _build_extractors(with_timestamp):
    # Factories keep construction lazy (per combo).
    return {
        "no_memory": NoMemory,
        "append_all": lambda: AppendAll(with_timestamp=with_timestamp),
        "regex": lambda: Regex(with_timestamp=with_timestamp),
        "regex_v2": lambda: Regex(version="v2", with_timestamp=with_timestamp),
        "ner": lambda: NER(with_timestamp=with_timestamp),
    }


def _dump(out_dir, name, preds):
    """Write per-question records for one combo to <out_dir>/<name>.jsonl."""
    with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps({
                "conv_id": p.qa_item.conv_id,
                "category": p.qa_item.category,
                "question": p.qa_item.question,
                "gold": p.qa_item.gold_answer,
                "answer": p.answer_text,
                "correct": p.judge_label,
                "judge_score": p.judge_score,
                "exact_match": exact_match(p.answer_text, p.qa_item.gold_answer),
                "token_f1": token_f1(p.answer_text, p.qa_item.gold_answer),
                "gold_evidence_dia_ids": p.qa_item.evidence_dia_ids,
                "retrieved_dia_ids": source_dia_ids(p.retrieved_memories),
                "retrieved_memories": [
                    {
                        "text": memory.text,
                        "source_dia_ids": memory.source_dia_ids,
                    }
                    for memory in p.retrieved_memories
                ],
            }, ensure_ascii=False) + "\n")


def _combos(extractors):
    for ename in extractors:
        for rname in RETRIEVERS:
            # no_memory pairs only with no_retrieval; real methods with real methods
            if (ename == "no_memory") != (rname == "no_retrieval"):
                continue
            yield ename, rname


def _free_gpu():
    """Release GPU memory after a model is dropped."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=None,
                    help="run only the first N dialogues; default all")
    variants = ap.add_mutually_exclusive_group()
    variants.add_argument("--base-only", action="store_true",
                          help="run only the base grid (default runs base and timestamp grids)")
    variants.add_argument("--timestamp", action="store_true",
                          help="run only the timestamp-augmented grid")
    ap.add_argument("--name", default=None,
                    help="output subfolder under results/; default is a timestamp")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the figure and table generation step")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="how many questions the reader and judge process at once")
    ap.add_argument("--combos", default=None,
                    help="comma-separated combos to run, e.g. append_all+tfidf,ner+bm25; default all")
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    dialogues, qa = load_locomo()
    if args.subset:
        dialogues = dialogues[: args.subset]

    if args.base_only:
        experiment_variants = [("base", False)]
    elif args.timestamp:
        experiment_variants = [("timestamp", True)]
    else:
        experiment_variants = [("base", False), ("timestamp", True)]

    selected = set(args.combos.split(",")) if args.combos else None
    planned = []
    for variant_name, with_timestamp in experiment_variants:
        extractors = _build_extractors(with_timestamp)
        for ename, rname in _combos(extractors):
            combo_name = f"{ename}+{rname}"
            if selected is not None and combo_name not in selected:
                continue
            # Timestamp augmentation has no effect on the no-memory baseline,
            # so score that baseline only once in the base grid.
            if with_timestamp and ename == "no_memory":
                continue
            planned.append((variant_name, combo_name, extractors[ename], RETRIEVERS[rname]))

    # Pass 1: reader answers every question
    try:
        reader = LocalLLMReader()
    except Exception as e:
        print(f"reader unavailable: {type(e).__name__}: {e}")
        return
    print(f"Pass 1: reader answering {len(planned)} configurations", flush=True)
    staged = []  # list of variant, combo name, records
    for i, (variant_name, combo_name, extractor_factory, retriever_factory) in enumerate(planned, 1):
        display_name = f"{variant_name}:{combo_name}"
        print(f"  [read {i}/{len(planned)}] {display_name}", flush=True)
        try:
            extractor, retriever = extractor_factory(), retriever_factory()
            records = read_pass(dialogues, qa, extractor, retriever, reader, K, args.batch_size)
        except NotImplementedError:
            print(f"  [skip] {display_name}: not implemented", flush=True)
            continue
        except Exception as e:
            print(f"  [skip] {display_name}: {type(e).__name__}: {e}", flush=True)
            continue
        staged.append((variant_name, combo_name, records))
    del reader
    _free_gpu()

    if not staged:
        print("nothing to score: implement at least one extractor and retriever")
        return

    # Pass 2: judge scores every answer
    try:
        judge = LocalLLMJudge()
    except Exception as e:
        print(f"judge unavailable: {type(e).__name__}: {e}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pass 2: judge scoring {len(staged)} configurations", flush=True)
    rows_by_variant = {name: [] for name, _ in experiment_variants}
    for i, (variant_name, combo_name, records) in enumerate(staged, 1):
        display_name = f"{variant_name}:{combo_name}"
        print(f"  [judge {i}/{len(staged)}] {display_name}", flush=True)
        try:
            preds = judge_pass(records, judge, args.batch_size)
            variant_dir = out_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            _dump(variant_dir, combo_name, preds)
            rows_by_variant[variant_name].append((combo_name, summarize(preds)))
        except Exception as e:
            print(f"  [skip] {display_name}: {type(e).__name__}: {e}", flush=True)
    del judge
    _free_gpu()

    # Write separate reports so existing analysis code can continue treating a
    # configuration name as exactly <extractor>+<retriever>.
    for variant_name, _ in experiment_variants:
        rows = rows_by_variant[variant_name]
        if not rows:
            continue
        variant_dir = out_dir / variant_name
        write_reports(rows, variant_dir, K)
        print(f"\n{variant_name.upper()}\n{render_console(rows, K)}")

        if not args.no_figures:
            try:
                from analyze import generate
                generate(variant_dir)
                print(f"Figures and tables written to {variant_dir / 'figures'}")
            except Exception as e:
                print(f"(figures skipped for {variant_name}: {type(e).__name__}: {e})")

    print(f"\nPer-question records and reports written under {out_dir}")


if __name__ == "__main__":
    main()
