"""Run the extract x retrieve combinations on LOCOMO and print QA accuracy.

Two passes keep one model in memory at a time: load the reader and answer all
questions, then load the judge and score all answers.
"""
import argparse
import json
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
from src.retrieve.word2vec import Word2vec
from src.retrieve.sentence_emb import SentenceEmb
from src.reader import LocalLLMReader
from src.judge import LocalLLMJudge

EXTRACTORS = {"no_memory": NoMemory, "append_all": AppendAll, "regex": Regex, "ner": NER}
RETRIEVERS = {"no_retrieval": NoRetrieval, "tfidf": Tfidf, "word2vec": Word2vec, "sentence_emb": SentenceEmb}
K = 5
RESULTS_DIR = Path(__file__).parent / "results"


def _dump(name, preds):
    """Write per-question records for one combo to results/<name>.jsonl."""
    with open(RESULTS_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
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


def _combos():
    for ename in EXTRACTORS:
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
    args = ap.parse_args()

    dialogues, qa = load_locomo()
    if args.subset:
        dialogues = dialogues[: args.subset]

    # Pass 1: reader answers every question
    try:
        reader = LocalLLMReader()
    except Exception as e:
        print(f"reader unavailable: {type(e).__name__}: {e}")
        return
    staged = []  # list of name, records
    for ename, rname in _combos():
        name = f"{ename}+{rname}"
        try:
            records = read_pass(dialogues, qa, EXTRACTORS[ename](), RETRIEVERS[rname](), reader, K)
        except NotImplementedError:
            print(f"[skip] {name}: method not implemented")
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
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for name, records in staged:
        preds = judge_pass(records, judge)
        _dump(name, preds)
        rows.append((name, summarize(preds)))
    del judge
    _free_gpu()

    # Reports
    write_reports(rows, RESULTS_DIR, K)
    print(f"\n{render_console(rows, K)}")
    print(f"\nPer-question records and reports written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
