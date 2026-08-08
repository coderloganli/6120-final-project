"""Run one round of the experiment: extract x retrieve combinations on LOCOMO,
then generate figures and tables.

Two passes keep one model in memory at a time: load the reader and answer all
questions, then load the judge and score all answers. Results go to a fresh
results/<name> folder; analyze.generate then writes figures and tables there.
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
from src.retrieve.sentence_emb import SentenceEmb, MODEL as SENT_MODEL
from src.reader import LocalLLMReader
from src.judge import LocalLLMJudge

# Heavy retriever models are loaded once and reused across all combos.
_shared = {}


def _w2v_vectors():
    if "w2v" not in _shared:
        import gensim.downloader
        _shared["w2v"] = gensim.downloader.load(W2V_MODEL)
    return _shared["w2v"]


def _sent_encoder():
    if "sent" not in _shared:
        from sentence_transformers import SentenceTransformer
        _shared["sent"] = SentenceTransformer(SENT_MODEL)
    return _shared["sent"]


RETRIEVERS = {
    "no_retrieval": NoRetrieval,
    "tfidf": Tfidf,
    "bm25": Bm25,
    "word2vec": lambda: Word2vec(keyed_vectors=_w2v_vectors()),                        # uniform pooling
    "word2vec_tfidf": lambda: Word2vec(keyed_vectors=_w2v_vectors(), weighting="tfidf"),
    "sentence_emb": lambda: SentenceEmb(model_name=SENT_MODEL, encoder=_sent_encoder()),
}
K = 5
RESULTS_ROOT = Path(__file__).parent / "results"


def _build_extractors(with_timestamp):
    # Factories keep construction lazy (per combo). --timestamp applies to all.
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
    ap.add_argument("--timestamp", action="store_true",
                    help="prepend the session date to each memory")
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
    extractors = _build_extractors(args.timestamp)
    dialogues, qa = load_locomo()
    if args.subset:
        dialogues = dialogues[: args.subset]

    # Pass 1: reader answers every question
    try:
        reader = LocalLLMReader()
    except Exception as e:
        print(f"reader unavailable: {type(e).__name__}: {e}")
        return
    combos = list(_combos(extractors))
    if args.combos:
        wanted = set(args.combos.split(","))
        combos = [(e, r) for e, r in combos if f"{e}+{r}" in wanted]
    print(f"Pass 1: reader answering {len(combos)} combos", flush=True)
    staged = []  # list of name, records
    for i, (ename, rname) in enumerate(combos, 1):
        name = f"{ename}+{rname}"
        print(f"  [read {i}/{len(combos)}] {name}", flush=True)
        try:
            extractor, retriever = extractors[ename](), RETRIEVERS[rname]()
            records = read_pass(dialogues, qa, extractor, retriever, reader, K, args.batch_size)
        except NotImplementedError:
            print(f"  [skip] {name}: not implemented", flush=True)
            continue
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
            continue
        staged.append((name, records))
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
    print(f"Pass 2: judge scoring {len(staged)} combos", flush=True)
    rows = []
    for i, (name, records) in enumerate(staged, 1):
        print(f"  [judge {i}/{len(staged)}] {name}", flush=True)
        try:
            preds = judge_pass(records, judge, args.batch_size)
            _dump(out_dir, name, preds)
            rows.append((name, summarize(preds)))
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
    del judge
    _free_gpu()

    # Reports
    write_reports(rows, out_dir, K)
    print(f"\n{render_console(rows, K)}")
    print(f"\nPer-question records and reports written to {out_dir}")

    # Figures and tables
    if not args.no_figures:
        try:
            from analyze import generate
            generate(out_dir)
            print(f"Figures and tables written to {out_dir / 'figures'}")
        except Exception as e:
            print(f"(figures skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
