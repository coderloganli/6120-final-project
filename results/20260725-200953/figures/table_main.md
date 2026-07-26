| Extractor \ Retriever | tfidf | bm25 | word2vec | word2vec_tfidf | sentence_emb |
| --- | --- | --- | --- | --- | --- |
| append_all | 45.5 | 47.3 | 44.5 | 49.6 | 38.8 |
| regex | 40.5 | 40.5 | 42.4 | 44.2 | 38.0 |
| regex_v2 | 42.8 | 43.8 | 44.4 | 46.4 | 40.3 |
| ner | 37.0 | 36.9 | 36.8 | 38.2 | 33.0 |

Baseline (no memory): 0.8

Cells are QA accuracy (%) judged by the LLM judge.