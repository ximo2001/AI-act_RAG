# Evaluation results

### Retrieval (32 answerable questions, unit-level Hit@k)

| Embedder | Chunk words | Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Gold in LLM context (top-5 chunks) | Hit@5 EN | Hit@5 ES | ms/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bge-small-en-v1.5 | 150 | bm25 | 46.9% | 75.0% | 81.2% | 87.5% | 0.622 | 78.1% | 92.6% | 20.0% | 1 |
| bge-small-en-v1.5 | 150 | dense | 50.0% | 65.6% | 71.9% | 78.1% | 0.603 | 68.8% | 81.5% | 20.0% | 13 |
| bge-small-en-v1.5 | 150 | hybrid | 56.2% | 75.0% | 78.1% | 87.5% | 0.672 | 78.1% | 88.9% | 20.0% | 15 |
| bge-small-en-v1.5 | 150 | hybrid+rerank | 56.2% | 84.4% | 90.6% | 90.6% | 0.719 | 90.6% | 100.0% | 40.0% | 515 |
| bge-small-en-v1.5 | 150 | hybrid+rerank+translate | 71.9% | 100.0% | 100.0% | 100.0% | 0.859 | 100.0% | 100.0% | 100.0% | 747 |
| bge-small-en-v1.5 | 400 | bm25 | 37.5% | 71.9% | 78.1% | 81.2% | 0.559 | 75.0% | 92.6% | 0.0% | 1 |
| bge-small-en-v1.5 | 400 | dense | 46.9% | 65.6% | 68.8% | 78.1% | 0.582 | 68.8% | 77.8% | 20.0% | 15 |
| bge-small-en-v1.5 | 400 | hybrid | 53.1% | 71.9% | 71.9% | 84.4% | 0.637 | 71.9% | 81.5% | 20.0% | 15 |
| bge-small-en-v1.5 | 400 | hybrid+rerank | 50.0% | 75.0% | 81.2% | 87.5% | 0.648 | 81.2% | 92.6% | 20.0% | 1270 |
| bge-small-en-v1.5 | 400 | hybrid+rerank+translate | 56.2% | 84.4% | 87.5% | 100.0% | 0.727 | 87.5% | 92.6% | 60.0% | 1509 |
| paraphrase-multilingual-MiniLM-L12-v2 | 150 | bm25 | 46.9% | 75.0% | 81.2% | 87.5% | 0.622 | 78.1% | 92.6% | 20.0% | 1 |
| paraphrase-multilingual-MiniLM-L12-v2 | 150 | dense | 40.6% | 50.0% | 62.5% | 75.0% | 0.492 | 56.2% | 63.0% | 60.0% | 14 |
| paraphrase-multilingual-MiniLM-L12-v2 | 150 | hybrid | 43.8% | 59.4% | 75.0% | 93.8% | 0.568 | 68.8% | 77.8% | 60.0% | 15 |
| paraphrase-multilingual-MiniLM-L12-v2 | 150 | hybrid+rerank | 59.4% | 90.6% | 90.6% | 93.8% | 0.748 | 90.6% | 100.0% | 40.0% | 521 |
| paraphrase-multilingual-MiniLM-L12-v2 | 150 | hybrid+rerank+translate | 75.0% | 100.0% | 100.0% | 100.0% | 0.875 | 100.0% | 100.0% | 100.0% | 755 |
| paraphrase-multilingual-MiniLM-L12-v2 | 400 | bm25 | 37.5% | 71.9% | 78.1% | 81.2% | 0.559 | 75.0% | 92.6% | 0.0% | 1 |
| paraphrase-multilingual-MiniLM-L12-v2 | 400 | dense | 6.2% | 12.5% | 15.6% | 21.9% | 0.103 | 15.6% | 18.5% | 0.0% | 12 |
| paraphrase-multilingual-MiniLM-L12-v2 | 400 | hybrid | 12.5% | 21.9% | 43.8% | 75.0% | 0.254 | 43.8% | 51.9% | 0.0% | 12 |
| paraphrase-multilingual-MiniLM-L12-v2 | 400 | hybrid+rerank | 50.0% | 71.9% | 78.1% | 87.5% | 0.630 | 78.1% | 85.2% | 40.0% | 1224 |
| paraphrase-multilingual-MiniLM-L12-v2 | 400 | hybrid+rerank+translate | 56.2% | 78.1% | 81.2% | 96.9% | 0.699 | 81.2% | 85.2% | 60.0% | 1530 |

_Run 2026-11-25T23:11:58+00:00 on 4 CPU cores, Python 3.13.18._

### End-to-end answers (32 answerable questions, 8 not answerable from the text)

| LLM | Prompt | Retrieval | Correct citation | Citation precision (answered) | Citation recall (answered) | Correct 'not in text' | False 'not in text' | Parse errors | Yes/No correct | Key facts | Faithfulness (EN) | Numbers grounded | Latency mean (s) | Latency p50 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m.gguf | v2 | hybrid+rerank | 71.9% | 57.9% | 71.0% | 50.0% | 3.1% | 0.0% | 71.4% | 40.3% | 96.3% | 100.0% | 14.6 | 12.0 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | v1 | hybrid+rerank+translate | 75.0% | 47.0% | 71.0% | 50.0% | 3.1% | 2.5% | 57.1% | 69.4% | 69.2% | 100.0% | 14.2 | 11.8 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | v2 | hybrid+rerank+translate | 78.1% | 59.2% | 75.0% | 50.0% | 0.0% | 0.0% | 71.4% | 52.8% | 96.3% | 100.0% | 14.6 | 13.7 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | v3 | hybrid+rerank+translate | 84.4% | 62.6% | 79.7% | 50.0% | 0.0% | 2.5% | 71.4% | 44.4% | 74.1% | 96.1% | 16.3 | 13.9 |
| qwen2.5-3b-instruct-q4_k_m.gguf | v1 | hybrid+rerank+translate | 87.5% | 55.7% | 81.2% | 25.0% | 0.0% | 0.0% | 85.7% | 77.8% | 80.2% | 93.8% | 26.6 | 22.4 |
| qwen2.5-3b-instruct-q4_k_m.gguf | v2 | hybrid+rerank+translate | 81.2% | 73.0% | 88.9% | 100.0% | 15.6% | 0.0% | 14.3% | 65.3% | 95.7% | 98.8% | 25.5 | 26.9 |
| qwen2.5-3b-instruct-q4_k_m.gguf | v3 | hybrid+rerank+translate | 90.6% | 66.4% | 84.4% | 100.0% | 0.0% | 0.0% | 85.7% | 69.4% | 77.8% | 100.0% | 21.9 | 19.4 |

_llama_cpp, temperature 0, top-k = 5 chunks of 150 words; run on 4 CPU cores._

### Where the errors come from

| Run | Retrieval miss | Generation miss (gold retrieved, not cited) | False 'not in text' | Hallucination on unanswerable | Unparseable output |
|---|---|---|---|---|---|
| qwen2.5-1.5b-instruct-q4_k_m.gguf / v2 | 3 (q28, q29, q30) | 6 (q01, q02, q12, q14, q25, q32) | 0 | 4 (q34, q35, q37, q38) | 0 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf / v1 / translate | 0 | 7 (q07, q12, q13, q14, q15, q19, q32) | 0 | 4 (q33, q34, q35, q37) | 1 (q17) |
| qwen2.5-1.5b-instruct-q4_k_m.gguf / v2 / translate | 0 | 7 (q01, q02, q12, q14, q25, q29, q32) | 0 | 4 (q34, q35, q37, q38) | 0 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf / v3 / translate | 0 | 5 (q01, q15, q25, q29, q32) | 0 | 3 (q34, q35, q37) | 1 (q40) |
| qwen2.5-3b-instruct-q4_k_m.gguf / v1 / translate | 0 | 4 (q15, q19, q26, q32) | 0 | 6 (q33, q34, q35, q36, q37, q38) | 0 |
| qwen2.5-3b-instruct-q4_k_m.gguf / v2 / translate | 0 | 1 (q25) | 5 (q06, q07, q11, q12, q30) | 0 | 0 |
| qwen2.5-3b-instruct-q4_k_m.gguf / v3 / translate | 0 | 3 (q12, q22, q25) | 0 | 0 | 0 |
