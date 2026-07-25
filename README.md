# Conversational Agent with Memory: extract x retrieve on LOCOMO

## Structure

```
run.py                   Runs every extract x retrieve combination, prints QA accuracy.

locomo/
  locomo10.json          LOCOMO dataset.
  loader.py              Loads locomo10.json into Dialogue and QAItem objects.

src/
  schema.py              Data structures and the four stage interfaces:
                         Extractor, Retriever, Reader, Judge.
  pipeline.py            read_pass and judge_pass over the data.
  metrics.py             summarize: overall QA accuracy.

  extract/
    no_memory.py         NoMemory: stores nothing, the lower-bound baseline. Implemented.
    append_all.py        AppendAll: stores each dialogue turn as one memory. Implemented.
    regex.py             Regex. To implement.
    ner.py               NER. To implement.
  retrieve/
    no_retrieval.py      NoRetrieval: retrieves nothing, pairs with NoMemory. Implemented.
    tfidf.py             Tfidf: dependency-free cosine retrieval baseline. Implemented.
    word2vec.py          Word2vec. To implement.
    sentence_emb.py      SentenceEmb. To implement.
  reader.py              LocalLLMReader: local model. Implemented.
  judge.py               LocalLLMJudge: local model. Implemented.
```

The six unimplemented method files are skeletons: a class subclassing its
interface with `raise NotImplementedError`. Fill in the method body to implement
a method.

## Run

Check LOCOMO loading:

```bash
python -m locomo.loader
```

Run every combination, including the no_memory lower bound:

```bash
python run.py
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

`python run.py` writes three report levels under `results/`:

- `<extractor>+<retriever>.jsonl`: auditable per-question answers, scores, and
  retrieved evidence.
- `summary.json`: machine-readable aggregate metrics.
- `report.md`: presentation-ready answer quality, retrieval quality, and
  category comparison tables.

The terminal also prints a compact overall comparison table.
