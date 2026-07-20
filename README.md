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
    append_all.py        AppendAll. To implement.
    regex.py             Regex. To implement.
    ner.py               NER. To implement.
  retrieve/
    no_retrieval.py      NoRetrieval: retrieves nothing, pairs with NoMemory. Implemented.
    tfidf.py             Tfidf. To implement.
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
