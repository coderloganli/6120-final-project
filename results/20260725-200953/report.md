# Evaluation Report

Retrieval metrics use the top 5 memories. Evidence metrics are macro-averaged over questions that provide gold evidence.

## Overall comparison

### Answer quality

| Method | N | Judge Acc. | Mean Judge Score | Exact Match | Token F1 |
| --- | --- | --- | --- | --- | --- |
| no_memory+no_retrieval | 1540 | 0.8% | 0.008 | 0.2% | 0.4% |
| append_all+tfidf | 1540 | 45.5% | 0.455 | 9.4% | 23.7% |
| append_all+bm25 | 1540 | 47.3% | 0.473 | 9.7% | 25.0% |
| append_all+word2vec | 1540 | 44.5% | 0.445 | 8.9% | 23.8% |
| append_all+word2vec_tfidf | 1540 | 49.6% | 0.496 | 8.8% | 25.6% |
| append_all+sentence_emb | 1540 | 38.8% | 0.388 | 6.9% | 19.5% |
| regex+tfidf | 1540 | 40.5% | 0.405 | 7.3% | 21.2% |
| regex+bm25 | 1540 | 40.5% | 0.405 | 7.8% | 21.8% |
| regex+word2vec | 1540 | 42.4% | 0.424 | 7.3% | 21.9% |
| regex+word2vec_tfidf | 1540 | 44.2% | 0.442 | 7.3% | 22.3% |
| regex+sentence_emb | 1540 | 38.0% | 0.380 | 5.8% | 18.8% |
| regex_v2+tfidf | 1540 | 42.8% | 0.428 | 8.8% | 22.9% |
| regex_v2+bm25 | 1540 | 43.8% | 0.438 | 8.8% | 23.4% |
| regex_v2+word2vec | 1540 | 44.4% | 0.444 | 8.8% | 23.5% |
| regex_v2+word2vec_tfidf | 1540 | 46.4% | 0.464 | 8.6% | 24.0% |
| regex_v2+sentence_emb | 1540 | 40.3% | 0.403 | 6.4% | 19.6% |
| ner+tfidf | 1540 | 37.0% | 0.370 | 6.8% | 18.8% |
| ner+bm25 | 1540 | 36.9% | 0.369 | 6.9% | 18.9% |
| ner+word2vec | 1540 | 36.8% | 0.368 | 6.4% | 18.6% |
| ner+word2vec_tfidf | 1540 | 38.2% | 0.382 | 6.8% | 20.1% |
| ner+sentence_emb | 1540 | 33.0% | 0.330 | 6.2% | 17.1% |

### Retrieval quality

| Method | Evidence N | Evidence Hit@5 | Evidence Precision@5 | Evidence Recall@5 | Full Recall@5 | MRR | Avg. Memories | Empty Retrieval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_memory+no_retrieval | 1536 | 0.0% | 0.0% | 0.0% | 0.0% | 0.000 | 0.00 | 100.0% |
| append_all+tfidf | 1536 | 53.2% | 11.4% | 48.2% | 44.3% | 0.375 | 4.99 | 0.0% |
| append_all+bm25 | 1536 | 55.9% | 11.9% | 50.5% | 46.4% | 0.408 | 4.99 | 0.0% |
| append_all+word2vec | 1536 | 52.2% | 11.3% | 45.8% | 41.1% | 0.357 | 5.00 | 0.0% |
| append_all+word2vec_tfidf | 1536 | 58.9% | 12.9% | 51.6% | 46.1% | 0.417 | 5.00 | 0.0% |
| append_all+sentence_emb | 1536 | 42.2% | 9.2% | 36.7% | 32.6% | 0.274 | 5.00 | 0.0% |
| regex+tfidf | 1536 | 50.0% | 10.8% | 44.4% | 40.2% | 0.391 | 4.99 | 0.1% |
| regex+bm25 | 1536 | 50.8% | 11.1% | 45.3% | 41.2% | 0.401 | 4.99 | 0.1% |
| regex+word2vec | 1536 | 51.6% | 11.4% | 44.3% | 38.8% | 0.364 | 5.00 | 0.0% |
| regex+word2vec_tfidf | 1536 | 53.5% | 11.9% | 46.2% | 40.8% | 0.404 | 5.00 | 0.0% |
| regex+sentence_emb | 1536 | 42.7% | 9.5% | 36.9% | 32.6% | 0.306 | 5.00 | 0.0% |
| regex_v2+tfidf | 1536 | 52.9% | 11.5% | 47.1% | 42.8% | 0.408 | 4.99 | 0.1% |
| regex_v2+bm25 | 1536 | 53.7% | 11.7% | 48.0% | 43.6% | 0.421 | 4.99 | 0.1% |
| regex_v2+word2vec | 1536 | 53.6% | 11.8% | 46.5% | 41.1% | 0.380 | 5.00 | 0.0% |
| regex_v2+word2vec_tfidf | 1536 | 55.9% | 12.4% | 48.8% | 43.3% | 0.423 | 5.00 | 0.0% |
| regex_v2+sentence_emb | 1536 | 45.0% | 10.0% | 38.8% | 34.2% | 0.318 | 5.00 | 0.0% |
| ner+tfidf | 1536 | 43.8% | 9.5% | 38.2% | 34.0% | 0.325 | 4.99 | 0.0% |
| ner+bm25 | 1536 | 43.7% | 9.6% | 38.4% | 34.3% | 0.340 | 4.99 | 0.0% |
| ner+word2vec | 1536 | 41.9% | 9.1% | 35.8% | 31.2% | 0.296 | 5.00 | 0.0% |
| ner+word2vec_tfidf | 1536 | 46.9% | 10.4% | 40.1% | 35.1% | 0.351 | 5.00 | 0.0% |
| ner+sentence_emb | 1536 | 36.5% | 8.1% | 31.2% | 27.2% | 0.264 | 5.00 | 0.0% |

## LOCOMO category breakdown

| Method | Category | N | Judge Acc. | Token F1 | Evidence Hit@5 | Evidence Recall@5 |
| --- | --- | --- | --- | --- | --- | --- |
| no_memory+no_retrieval | 1: Multi-hop | 282 | 1.4% | 0.2% | 0.0% | 0.0% |
| no_memory+no_retrieval | 2: Temporal | 321 | 0.0% | 0.0% | 0.0% | 0.0% |
| no_memory+no_retrieval | 3: Open-domain | 96 | 6.2% | 4.2% | 0.0% | 0.0% |
| no_memory+no_retrieval | 4: Single-hop | 841 | 0.4% | 0.2% | 0.0% | 0.0% |
| append_all+tfidf | 1: Multi-hop | 282 | 38.7% | 14.5% | 34.8% | 17.0% |
| append_all+tfidf | 2: Temporal | 321 | 18.4% | 9.3% | 63.2% | 60.2% |
| append_all+tfidf | 3: Open-domain | 96 | 28.1% | 8.6% | 31.5% | 23.3% |
| append_all+tfidf | 4: Single-hop | 841 | 60.0% | 34.1% | 57.9% | 56.8% |
| append_all+bm25 | 1: Multi-hop | 282 | 40.1% | 15.9% | 37.9% | 18.4% |
| append_all+bm25 | 2: Temporal | 321 | 18.1% | 9.7% | 63.6% | 60.6% |
| append_all+bm25 | 3: Open-domain | 96 | 34.4% | 9.3% | 32.6% | 23.3% |
| append_all+bm25 | 4: Single-hop | 841 | 62.3% | 35.6% | 61.6% | 60.4% |
| append_all+word2vec | 1: Multi-hop | 282 | 46.5% | 18.6% | 48.2% | 22.6% |
| append_all+word2vec | 2: Temporal | 321 | 12.8% | 6.8% | 51.1% | 48.7% |
| append_all+word2vec | 3: Open-domain | 96 | 27.1% | 11.8% | 30.4% | 20.5% |
| append_all+word2vec | 4: Single-hop | 841 | 58.0% | 33.5% | 56.4% | 55.2% |
| append_all+word2vec_tfidf | 1: Multi-hop | 282 | 56.7% | 21.2% | 56.7% | 27.8% |
| append_all+word2vec_tfidf | 2: Temporal | 321 | 14.6% | 8.9% | 64.8% | 62.0% |
| append_all+word2vec_tfidf | 3: Open-domain | 96 | 29.2% | 10.2% | 38.0% | 28.1% |
| append_all+word2vec_tfidf | 4: Single-hop | 841 | 62.9% | 35.3% | 59.7% | 58.2% |
| append_all+sentence_emb | 1: Multi-hop | 282 | 38.3% | 13.9% | 39.0% | 19.0% |
| append_all+sentence_emb | 2: Temporal | 321 | 10.3% | 6.5% | 44.5% | 41.5% |
| append_all+sentence_emb | 3: Open-domain | 96 | 38.5% | 11.4% | 27.2% | 18.6% |
| append_all+sentence_emb | 4: Single-hop | 841 | 49.8% | 27.3% | 44.0% | 42.7% |
| regex+tfidf | 1: Multi-hop | 282 | 41.5% | 16.7% | 44.0% | 22.0% |
| regex+tfidf | 2: Temporal | 321 | 15.0% | 8.6% | 65.4% | 62.6% |
| regex+tfidf | 3: Open-domain | 96 | 33.3% | 11.8% | 26.1% | 17.7% |
| regex+tfidf | 4: Single-hop | 841 | 50.8% | 28.6% | 48.8% | 47.8% |
| regex+bm25 | 1: Multi-hop | 282 | 40.8% | 16.9% | 42.9% | 22.3% |
| regex+bm25 | 2: Temporal | 321 | 15.3% | 8.9% | 65.4% | 62.7% |
| regex+bm25 | 3: Open-domain | 96 | 34.4% | 12.2% | 27.2% | 18.9% |
| regex+bm25 | 4: Single-hop | 841 | 50.8% | 29.5% | 50.4% | 49.3% |
| regex+word2vec | 1: Multi-hop | 282 | 52.1% | 21.3% | 56.7% | 28.4% |
| regex+word2vec | 2: Temporal | 321 | 13.7% | 8.6% | 57.9% | 55.1% |
| regex+word2vec | 3: Open-domain | 96 | 29.2% | 11.0% | 33.7% | 23.2% |
| regex+word2vec | 4: Single-hop | 841 | 51.6% | 28.4% | 49.3% | 47.9% |
| regex+word2vec_tfidf | 1: Multi-hop | 282 | 50.7% | 20.2% | 57.1% | 29.3% |
| regex+word2vec_tfidf | 2: Temporal | 321 | 16.2% | 9.2% | 64.8% | 62.2% |
| regex+word2vec_tfidf | 3: Open-domain | 96 | 34.4% | 12.3% | 37.0% | 24.8% |
| regex+word2vec_tfidf | 4: Single-hop | 841 | 53.9% | 29.1% | 49.7% | 48.1% |
| regex+sentence_emb | 1: Multi-hop | 282 | 42.6% | 17.2% | 44.3% | 22.3% |
| regex+sentence_emb | 2: Temporal | 321 | 14.6% | 7.3% | 53.9% | 50.5% |
| regex+sentence_emb | 3: Open-domain | 96 | 37.5% | 12.6% | 27.2% | 18.5% |
| regex+sentence_emb | 4: Single-hop | 841 | 45.4% | 24.4% | 39.6% | 38.6% |
| regex_v2+tfidf | 1: Multi-hop | 282 | 42.6% | 17.0% | 45.4% | 23.3% |
| regex_v2+tfidf | 2: Temporal | 321 | 16.8% | 9.1% | 65.4% | 62.3% |
| regex_v2+tfidf | 3: Open-domain | 96 | 28.1% | 9.5% | 29.3% | 21.0% |
| regex_v2+tfidf | 4: Single-hop | 841 | 54.5% | 31.6% | 53.2% | 52.1% |
| regex_v2+bm25 | 1: Multi-hop | 282 | 42.6% | 17.7% | 44.7% | 23.4% |
| regex_v2+bm25 | 2: Temporal | 321 | 15.6% | 8.9% | 65.7% | 62.6% |
| regex_v2+bm25 | 3: Open-domain | 96 | 33.3% | 10.1% | 29.3% | 20.7% |
| regex_v2+bm25 | 4: Single-hop | 841 | 56.1% | 32.4% | 54.8% | 53.6% |
| regex_v2+word2vec | 1: Multi-hop | 282 | 50.4% | 20.6% | 55.3% | 27.6% |
| regex_v2+word2vec | 2: Temporal | 321 | 14.6% | 8.2% | 57.6% | 54.8% |
| regex_v2+word2vec | 3: Open-domain | 96 | 29.2% | 11.7% | 34.8% | 24.4% |
| regex_v2+word2vec | 4: Single-hop | 841 | 55.5% | 31.7% | 53.5% | 52.1% |
| regex_v2+word2vec_tfidf | 1: Multi-hop | 282 | 51.8% | 19.8% | 57.1% | 29.5% |
| regex_v2+word2vec_tfidf | 2: Temporal | 321 | 16.8% | 9.5% | 65.7% | 63.4% |
| regex_v2+word2vec_tfidf | 3: Open-domain | 96 | 34.4% | 12.7% | 39.1% | 28.5% |
| regex_v2+word2vec_tfidf | 4: Single-hop | 841 | 57.3% | 32.1% | 53.5% | 52.0% |
| regex_v2+sentence_emb | 1: Multi-hop | 282 | 48.6% | 17.1% | 47.5% | 23.8% |
| regex_v2+sentence_emb | 2: Temporal | 321 | 14.3% | 6.9% | 53.6% | 50.0% |
| regex_v2+sentence_emb | 3: Open-domain | 96 | 35.4% | 11.7% | 30.4% | 21.2% |
| regex_v2+sentence_emb | 4: Single-hop | 841 | 47.9% | 26.2% | 42.4% | 41.5% |
| ner+tfidf | 1: Multi-hop | 282 | 38.3% | 14.3% | 39.0% | 18.5% |
| ner+tfidf | 2: Temporal | 321 | 17.1% | 9.6% | 64.8% | 61.3% |
| ner+tfidf | 3: Open-domain | 96 | 29.2% | 8.9% | 25.0% | 16.1% |
| ner+tfidf | 4: Single-hop | 841 | 45.1% | 24.9% | 39.5% | 38.4% |
| ner+bm25 | 1: Multi-hop | 282 | 40.1% | 15.9% | 37.6% | 18.7% |
| ner+bm25 | 2: Temporal | 321 | 16.5% | 9.2% | 64.5% | 61.1% |
| ner+bm25 | 3: Open-domain | 96 | 29.2% | 8.3% | 23.9% | 15.8% |
| ner+bm25 | 4: Single-hop | 841 | 44.5% | 24.8% | 40.0% | 38.8% |
| ner+word2vec | 1: Multi-hop | 282 | 42.9% | 16.5% | 46.1% | 21.1% |
| ner+word2vec | 2: Temporal | 321 | 15.6% | 8.2% | 56.7% | 54.2% |
| ner+word2vec | 3: Open-domain | 96 | 25.0% | 10.5% | 18.5% | 13.8% |
| ner+word2vec | 4: Single-hop | 841 | 44.2% | 24.3% | 37.5% | 36.0% |
| ner+word2vec_tfidf | 1: Multi-hop | 282 | 45.7% | 18.6% | 52.5% | 25.1% |
| ner+word2vec_tfidf | 2: Temporal | 321 | 16.2% | 10.0% | 64.8% | 62.7% |
| ner+word2vec_tfidf | 3: Open-domain | 96 | 28.1% | 10.3% | 32.6% | 23.0% |
| ner+word2vec_tfidf | 4: Single-hop | 841 | 45.2% | 25.5% | 39.8% | 38.4% |
| ner+sentence_emb | 1: Multi-hop | 282 | 39.7% | 15.2% | 37.9% | 18.4% |
| ner+sentence_emb | 2: Temporal | 321 | 16.2% | 7.9% | 55.1% | 51.5% |
| ner+sentence_emb | 3: Open-domain | 96 | 29.2% | 11.1% | 21.7% | 14.2% |
| ner+sentence_emb | 4: Single-hop | 841 | 37.6% | 21.9% | 30.6% | 29.6% |

Categories: 1 = Multi-hop, 2 = Temporal, 3 = Open-domain, 4 = Single-hop.
