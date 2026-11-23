# MeshRAG

Code accompanying [Collision to Cognition: Hash-Driven Graph Construction for Efficient RAG](https://aclanthology.org/2026.acl-long.1156.pdf) (ACL 2026).

## Hash graph construction

MeshRAG organizes corpus passages into a two-level graph using locality-sensitive hashing. Chunks sharing hash buckets form candidate neighbors, and their closest hash neighbors become chunk-level edges. Collision buckets also seed overlapping communities, which are consolidated by passage overlap. A passage may belong to multiple communities.


## Run on HotpotQA

Use Python 3.10 or newer. Install the dependencies and set your API key:

```bash
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY='your-key'
```

The question set and corpus are included in `dataset/`. Run the first 200 questions with:

```bash
python benchmark.py --limit 200 \
  --encoder sentence-transformers/all-MiniLM-L6-v2
```

Answer a single question with:

```bash
python main.py --corpus dataset/hotpotqa_corpus.json \
  --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --question 'Who directed Casablanca?'
```

The encoder is downloaded on first use. To use a local encoder, pass its directory through `--encoder`. Generation and LLM evaluation use `deepseek-flash` by default; `--model` selects another model. Set `DEEPSEEK_BASE_URL` if a custom API endpoint is needed.

The benchmark saves answers, evaluations, retrieval traces and candidate-pool statistics under `runs/`. Repeating the same command resumes saved progress. Corpus embeddings and the hash graph are cached under `state/`.

An example run on the first 200 questions achieved **67.0% ContainsMatch** and **81.0% LLM accuracy**. Across 509 retrieval calls, BERT reranked an average of **253.34 out of 9,811 passages (2.58%)**. A compact summary is available in [results/hotpotqa_200.json](results/hotpotqa_200.json).

Run the tests with:

```bash
python -m unittest discover -v
```
