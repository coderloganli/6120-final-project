# Conversational Agent with Memory: extract x retrieve on LOCOMO

## Structure

```
demo.py                   Runs every extract x retrieve combination, then writes figures and tables.
analyze.py               Regenerates figures and tables from a saved run, without rerunning models.

locomo/
  locomo10.json          LOCOMO dataset.
  loader.py              Loads locomo10.json into Dialogue and QAItem objects.

src/
  schema.py              Data structures and the four stage interfaces:
                         Extractor, Retriever, Reader, Judge.
  pipeline.py            read_pass and judge_pass over the data, in batches.
  metrics.py             summarize: answer, retrieval, and per-category metrics.
  report.py              Console table, Markdown, and JSON reports.

  extract/
    no_memory.py         NoMemory: stores nothing, the lower-bound baseline.
    append_all.py        AppendAll: stores each dialogue turn as one memory.
    regex.py             Regex: keeps turns matching hand-written patterns.
    ner.py               NER: keeps turns that mention a named entity.
    textfmt.py           Shared memory-text formatting.
  retrieve/
    no_retrieval.py      NoRetrieval: retrieves nothing, pairs with NoMemory.
    tfidf.py             Tfidf: TF-IDF cosine retrieval.
    bm25.py              Bm25: BM25 retrieval.
    word2vec.py          Word2vec: averaged static embeddings.
    sentence_emb.py      SentenceEmb: sentence-transformer embeddings.
  reader.py              LocalLLMReader: local model that answers questions.
  judge.py               LocalLLMJudge: local model that scores answers.

tests/                   Unit tests for extractors, retrievers, metrics, and reports.
```

Every method implements an interface in `schema.py`. To add a new extractor or
retriever, subclass its interface and register it in `run.py`.

## Setup

Create a virtual environment and install the dependencies. Use Python 3.11;
torch has no wheels for 3.14.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
```

Run everything with `.venv/bin/python`, for example `.venv/bin/python demo.py`.

Notes:

- The reader, judge, and word2vec models download on first use into the shared
  Hugging Face and gensim caches under your home directory, not into this repo.
  The first run pulls tens of GB; later runs reuse the cache.
- `.venv/` and `results/` are git-ignored and are not carried by `git clone` or a
  plain folder copy. In a new location, recreate the venv with the commands
  above. On the same machine the model cache is shared, so models are not
  re-downloaded.

## Run

Check LOCOMO loading:

```bash
python -m locomo.loader
```

Run every combination, including the no_memory lower bound:

```bash
python demo.py
```

Rows whose methods are not implemented are skipped. Use `--subset N` to run only
the first N dialogues.

## Basic memory baseline

`append_all+tfidf` is the first end-to-end memory baseline:

1. `AppendAll` converts every dialogue turn into one `Memory`. Its text includes
   the speaker and its metadata retains the session and timestamp.
2. `Tfidf` indexes those memories and returns the top `k` memories with positive
   cosine similarity to the question.
3. The reader answers from the retrieved turns and the judge scores the answer.

The implementation uses only the Python standard library; the local reader and
judge models remain the only heavyweight runtime dependencies.

Run the unit tests without loading a model:

```bash
python -m unittest discover -s tests -v
```

## Evaluation and reports

Each run reports complementary answer and retrieval metrics:

- **Judge accuracy / score:** semantic correctness from the local LLM judge.
- **Exact match and token F1:** deterministic lexical answer metrics after
  lowercasing and punctuation/article normalization.
- **Evidence Hit@K, Precision@K, Recall@K, full Recall@K, and MRR:** retrieval
  quality against LOCOMO evidence turn IDs. These are macro-averaged only over
  questions with gold evidence.
- **Average memories and empty retrieval rate:** context usage diagnostics over
  all questions.
- **Category breakdown:** the same metrics for multi-hop, temporal, open-domain,
  and single-hop questions.

`python demo.py` writes three report levels under `results/`:

- `<extractor>+<retriever>.jsonl`: auditable per-question answers, scores, and
  retrieved evidence.
- `summary.json`: machine-readable aggregate metrics.
- `report.md`: presentation-ready answer quality, retrieval quality, and
  category comparison tables.

The terminal also prints a compact overall comparison table.
