# RAG relevant vs irrelevant retrieval (design)

Date: 2026-05-10

## Summary
Add an "irrelevant RAG" mode that uses fixed Wikipedia pages to test whether retrieval changes model answers due to added context rather than relevance. Keep existing relevant RAG behavior over TOP_N_CHUNKS (0/1/3/5). Irrelevant RAG always runs top_k=3 and executes once per fixed URL. Output CSV gains `top_k`, `rag_relevante`, and `rag_url`.

## Goals
- Compare baseline (no RAG) vs relevant RAG vs irrelevant RAG.
- Keep relevant RAG on existing TOP_N_CHUNKS values.
- Run irrelevant RAG exactly 3 times per assertion, one for each fixed URL, with top_k=3.
- Ensure outputs identify the URL used for relevant and irrelevant contexts.
- Fail fast if a required URL is not indexed in wiki_faiss_store.

## Non-goals
- Changing how pairs are generated or validated.
- Adding new external data sources outside wiki_faiss_store.
- Re-ranking or changing embedding model behavior.

## Configuration
Add to config:
- `WIKI_IRRELEVANT_URLS`: list of 3 Wikipedia URLs (already indexed).

Keep:
- `TOP_N_CHUNKS`: relevant RAG values (0/1/3/5).

## Data outputs
Add CSV columns:
- `top_k` (int): 0 for baseline, 1/3/5 for relevant, 3 for irrelevant.
- `rag_relevante` (bool): true for relevant, false for irrelevant and baseline.
- `rag_url` (string):
  - relevant: `pair["wiki"]`
  - irrelevant: one of `WIKI_IRRELEVANT_URLS`
  - baseline: empty string

Existing columns remain unchanged.

## Retrieval behavior
### Relevant RAG
- Uses `pair["wiki"]` as the single page URL.
- For each `top_k` in `TOP_N_CHUNKS`:
  - `top_k == 0`: baseline, no retrieval, `rag_relevante=false`, `rag_url=""`.
  - `top_k > 0`: retrieval restricted to that URL, `rag_relevante=true`, `rag_url=pair["wiki"]`.

### Irrelevant RAG
- For each URL in `WIKI_IRRELEVANT_URLS`:
  - Always `top_k=3`.
  - Retrieval restricted to that URL.
  - `rag_relevante=false`, `rag_url=<url>`.

### URL filtering
- `WikiRetriever` gains `page_url` filtering.
- Retrieval only considers chunks from the matching URL.
- If the URL is missing in the store, raise a runtime error (no fallback).

## Error handling
- Pre-validate that all `pair["wiki"]` URLs and all `WIKI_IRRELEVANT_URLS` exist in wiki_faiss_store.
- If any required URL is missing, abort the run with a clear error.

## Cache keys
Include `rag_url` and `rag_relevante` in `cache_extra` to avoid mixing responses between modes.

## Testing (TDD)
- Retriever URL filter returns only chunks from the requested URL.
- Retriever raises when URL is missing.
- Run path builds task metadata with correct `top_k`, `rag_relevante`, `rag_url` for baseline, relevant, and irrelevant.

## Open questions
None.
