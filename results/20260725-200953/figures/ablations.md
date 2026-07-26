## Regex rules: v1 vs v2

| Retriever | v1 | v2 | delta |
| --- | --- | --- | --- |
| tfidf | 40.5 | 42.8 | +2.3 |
| bm25 | 40.5 | 43.8 | +3.2 |
| word2vec | 42.4 | 44.4 | +2.0 |
| word2vec_tfidf | 44.2 | 46.4 | +2.2 |
| sentence_emb | 38.0 | 40.3 | +2.3 |

## Word2vec pooling: uniform vs tfidf

| Extractor | uniform | tfidf | delta |
| --- | --- | --- | --- |
| append_all | 44.5 | 49.6 | +5.1 |
| regex | 42.4 | 44.2 | +1.8 |
| regex_v2 | 44.4 | 46.4 | +2.0 |
| ner | 36.8 | 38.2 | +1.4 |

Delta is percentage points (v2 minus v1; tfidf minus uniform).