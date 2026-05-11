# RAG Irrelevant Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add irrelevant RAG runs (top_k=3, one per fixed URL) alongside existing relevant RAG and baseline, and emit `top_k`, `rag_relevante`, `rag_url` in the output CSV.

**Architecture:** Extend the wiki retriever to filter by a specific page URL and fail fast when a URL is missing. Add pure helper functions in `main.py` to generate RAG modes and validate URLs, then wire those helpers into the task creation flow. Introduce unit tests (unittest) to enforce URL-filtered retrieval and RAG mode expansion.

**Tech Stack:** Python 3.12, Hydra, pandas, FAISS, numpy, unittest (stdlib)

---

## File Structure

**Modify**
- `src/main/wiki_retrieval.py`: add URL filtering, URL existence checks, and page_url support in retrieve/build_context.
- `main.py`: add RAG mode helpers, URL validation, and wire new modes into task generation; pass page_url into retrieval.
- `conf/config.yaml`: add `WIKI_IRRELEVANT_URLS` list.

**Create**
- `tests/test_wiki_retrieval.py`: URL-filter retrieval tests.
- `tests/test_main_helpers.py`: RAG mode and URL validation tests.

---

### Task 1: URL-filtered retrieval (WikiRetriever)

**Files:**
- Create: `tests/test_wiki_retrieval.py`
- Modify: `src/main/wiki_retrieval.py`

- [ ] **Step 1: Write the failing test (URL filter on retrieve)**

```python
import os
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from src.main.wiki_retrieval import WikiRetriever


class WikiRetrieverUrlFilterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        os.environ["DEEPINFRA_API_KEY"] = "test"
        self.tmp = tempfile.TemporaryDirectory()
        store = Path(self.tmp.name)

        self.url1 = "https://pt.wikipedia.org/wiki/Url1"
        self.url2 = "https://pt.wikipedia.org/wiki/Url2"

        docs = [
            {
                "doc_id": "d1",
                "page_key": "p1",
                "title": "Page 1",
                "url": self.url1,
                "chunk_id": 0,
                "text": "chunk 1",
            },
            {
                "doc_id": "d2",
                "page_key": "p1",
                "title": "Page 1",
                "url": self.url1,
                "chunk_id": 1,
                "text": "chunk 2",
            },
            {
                "doc_id": "d3",
                "page_key": "p2",
                "title": "Page 2",
                "url": self.url2,
                "chunk_id": 0,
                "text": "other",
            },
        ]

        with (store / "docs.jsonl").open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        np.save(store / "embeddings.npy", embeddings)

        self.retriever = WikiRetriever(store_dir=store, embedding_model="dummy")

        async def _fake_embed(_self, _text):
            return np.array([[1.0, 0.0]], dtype=np.float32)

        self.retriever._embed_query = types.MethodType(_fake_embed, self.retriever)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_retrieve_filters_by_url(self):
        chunks = await self.retriever.retrieve("query", top_n=2, page_url=self.url1)
        self.assertTrue(chunks)
        self.assertTrue(all(c.url == self.url1 for c in chunks))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: FAIL with `TypeError: retrieve() got an unexpected keyword argument 'page_url'`.

- [ ] **Step 3: Implement minimal URL-filtered retrieval**

Update `src/main/wiki_retrieval.py`:

```python
class WikiRetriever:
    def __init__(...):
        ...
        self._url_to_indices: dict[str, np.ndarray] = {}
        for idx, doc in enumerate(self._store.docs):
            url = str(doc.get("url", ""))
            if not url:
                continue
            self._url_to_indices.setdefault(url, []).append(idx)
        self._url_to_indices = {
            url: np.array(indices, dtype=np.int64)
            for url, indices in self._url_to_indices.items()
        }
        ...

    async def _retrieve_uncached(
        self,
        query: str,
        top_n: int,
        page_url: str | None = None,
    ) -> list[WikiChunk]:
        if top_n <= 0:
            return []

        if len(self._store.docs) == 0:
            raise RuntimeError("O índice está vazio (docs.jsonl sem entradas).")

        query_embedding = await self._embed_query(query)

        if page_url:
            indices = self._url_to_indices.get(page_url)
            if indices is None:
                raise RuntimeError(f"URL não encontrada no índice: {page_url}")

            subset = self._store.embeddings[indices]
            scores = (subset @ query_embedding.T).reshape(-1)
            k = min(int(top_n), int(scores.shape[0]))
            if k <= 0:
                return []

            top_local = np.argpartition(-scores, k - 1)[:k]
            top_local = top_local[np.argsort(-scores[top_local])]

            results: list[WikiChunk] = []
            for rank, local_idx in enumerate(top_local, start=1):
                global_idx = int(indices[int(local_idx)])
                doc = self._store.docs[global_idx]
                results.append(
                    WikiChunk(
                        rank=rank,
                        score=float(scores[int(local_idx)]),
                        doc_id=str(doc.get("doc_id", "")),
                        page_key=str(doc.get("page_key", "")),
                        title=str(doc.get("title", "")),
                        url=str(doc.get("url", "")),
                        chunk_id=int(doc.get("chunk_id", -1)),
                        text=str(doc.get("text", "")),
                    )
                )
            return results

        k = min(int(top_n), len(self._store.docs))
        scores, idxs = self._store.index.search(query_embedding, k)
        ...

    async def retrieve(self, query: str, *, top_n: int = 5, page_url: str | None = None) -> list[WikiChunk]:
        key = (query, int(top_n), page_url or "")

        async with self._cache_lock:
            fut = self._result_cache.get(key)
            if fut is None:
                fut = asyncio.create_task(
                    self._retrieve_uncached(query, int(top_n), page_url=page_url)
                )
                self._result_cache[key] = fut

        return await fut
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add tests/test_wiki_retrieval.py src/main/wiki_retrieval.py
git commit -m "test: add URL-filtered wiki retrieval"
```

---

### Task 2: Missing URL raises error

**Files:**
- Modify: `tests/test_wiki_retrieval.py`
- Modify: `src/main/wiki_retrieval.py`

- [ ] **Step 1: Add failing test for missing URL**

Append to `tests/test_wiki_retrieval.py`:

```python
    async def test_retrieve_raises_on_missing_url(self):
        with self.assertRaises(RuntimeError):
            await self.retriever.retrieve("query", top_n=1, page_url="https://missing")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: FAIL (no error raised).

- [ ] **Step 3: Implement the error on missing URL**

Ensure the `page_url` branch in `_retrieve_uncached` raises when missing:

```python
            indices = self._url_to_indices.get(page_url)
            if indices is None:
                raise RuntimeError(f"URL não encontrada no índice: {page_url}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_wiki_retrieval.py src/main/wiki_retrieval.py
git commit -m "test: raise on missing wiki URL"
```

---

### Task 3: build_context supports page_url

**Files:**
- Modify: `tests/test_wiki_retrieval.py`
- Modify: `src/main/wiki_retrieval.py`

- [ ] **Step 1: Add failing test for build_context page_url**

Append to `tests/test_wiki_retrieval.py`:

```python
    async def test_build_context_uses_page_url(self):
        ctx = await self.retriever.build_context("query", top_n=1, page_url=self.url2)
        self.assertIn(self.url2, ctx)
        self.assertNotIn(self.url1, ctx)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: FAIL with `TypeError: build_context() got an unexpected keyword argument 'page_url'`.

- [ ] **Step 3: Implement page_url in build_context**

Update `build_context`:

```python
    async def build_context(
        self,
        query: str,
        *,
        top_n: int = 5,
        max_chars_per_chunk: int = 9000,
        page_url: str | None = None,
    ) -> str:
        chunks = await self.retrieve(query, top_n=int(top_n), page_url=page_url)
        if not chunks:
            return ""

        blocks: list[str] = []
        for c in chunks:
            text = c.text
            if max_chars_per_chunk and len(text) > int(max_chars_per_chunk):
                text = text[: int(max_chars_per_chunk)] + "..."
            blocks.append("\n".join(["- " + text]))

        first = chunks[0]
        return "\n" + f"Página: {first.title}" + "\n" + f"URL: {first.url}" + "\n\n" + "\n\n".join(blocks)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_wiki_retrieval.py -v`

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_wiki_retrieval.py src/main/wiki_retrieval.py
git commit -m "feat: support page_url in build_context"
```

---

### Task 4: RAG mode helper

**Files:**
- Create: `tests/test_main_helpers.py`
- Modify: `main.py`

- [ ] **Step 1: Write failing test for build_rag_modes**

Create `tests/test_main_helpers.py` with:

```python
import unittest

from main import build_rag_modes


class RagModeTests(unittest.TestCase):
    def test_build_rag_modes_includes_irrelevant(self):
        top_n = [0, 1, 3, 5]
        irrelevant = ["u1", "u2", "u3"]

        modes = build_rag_modes(top_n, "wiki", irrelevant)

        self.assertEqual(len(modes), 7)
        self.assertEqual(modes[0]["top_k"], 0)
        self.assertEqual(modes[0]["rag_relevante"], False)
        self.assertEqual(modes[0]["rag_url"], "")

        self.assertEqual(modes[1]["top_k"], 1)
        self.assertEqual(modes[1]["rag_relevante"], True)
        self.assertEqual(modes[1]["rag_url"], "wiki")

        irrelevant_urls = [m["rag_url"] for m in modes if m["top_k"] == 3 and not m["rag_relevante"]]
        self.assertEqual(irrelevant_urls, irrelevant)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: FAIL with `ImportError` or `AttributeError` (build_rag_modes missing).

- [ ] **Step 3: Implement build_rag_modes**

Add to `main.py` (near top-level helpers):

```python
def build_rag_modes(top_n_chunks_list, pair_wiki_url, irrelevant_urls):
    modes = []
    for top_n in top_n_chunks_list:
        top_k = int(top_n or 0)
        if top_k <= 0:
            modes.append(
                {
                    "top_k": 0,
                    "rag_relevante": False,
                    "rag_url": "",
                    "page_url": None,
                }
            )
        else:
            modes.append(
                {
                    "top_k": top_k,
                    "rag_relevante": True,
                    "rag_url": pair_wiki_url,
                    "page_url": pair_wiki_url,
                }
            )

    for url in irrelevant_urls:
        modes.append(
            {
                "top_k": 3,
                "rag_relevante": False,
                "rag_url": url,
                "page_url": url,
            }
        )

    return modes
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add tests/test_main_helpers.py main.py
git commit -m "feat: add rag mode helper"
```

---

### Task 5: URL validation helper

**Files:**
- Modify: `tests/test_main_helpers.py`
- Modify: `src/main/wiki_retrieval.py`

- [ ] **Step 1: Add failing test for URL validation**

Append to `tests/test_main_helpers.py`:

```python
from src.main.wiki_retrieval import WikiRetriever
import json
import os
import tempfile
from pathlib import Path
import numpy as np


class RagUrlValidationTests(unittest.TestCase):
    def test_ensure_urls_raises_for_missing(self):
        os.environ["DEEPINFRA_API_KEY"] = "test"
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            docs = [
                {
                    "doc_id": "d1",
                    "page_key": "p1",
                    "title": "Page 1",
                    "url": "u1",
                    "chunk_id": 0,
                    "text": "x",
                }
            ]
            with (store / "docs.jsonl").open("w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
            np.save(store / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))

            retriever = WikiRetriever(store_dir=store, embedding_model="dummy")

            with self.assertRaises(RuntimeError):
                retriever.ensure_urls(["u1", "missing"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: FAIL (ensure_urls missing).

- [ ] **Step 3: Implement URL validation on WikiRetriever**

Add to `src/main/wiki_retrieval.py`:

```python
    def has_url(self, url: str) -> bool:
        return url in self._url_to_indices

    def ensure_urls(self, urls: list[str]) -> None:
        missing = [u for u in urls if u not in self._url_to_indices]
        if missing:
            raise RuntimeError(f"URL(s) não encontradas no índice: {missing}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_main_helpers.py src/main/wiki_retrieval.py
git commit -m "feat: validate wiki URLs exist"
```

---

### Task 6: Wire RAG modes into main.run and pass page_url

**Files:**
- Modify: `main.py`
- Modify: `conf/config.yaml`

- [ ] **Step 1: Add failing test for rag context expansion**

Append to `tests/test_main_helpers.py`:

```python
from main import iter_rag_contexts


class RagContextTests(unittest.TestCase):
    def test_iter_rag_contexts_emits_all_modes(self):
        pair = {"wiki": "wiki", "pair_id": 1, "eixo": "x", "p_plus": "p+", "p_minus": "p-"}
        top_n = [0, 1, 3, 5]
        irrelevant = ["u1", "u2", "u3"]

        contexts = list(iter_rag_contexts([pair], top_n, irrelevant))

        self.assertEqual(len(contexts), 7)
        self.assertTrue(any(c["top_k"] == 0 and c["rag_url"] == "" for c in contexts))
        self.assertTrue(any(c["top_k"] == 1 and c["rag_url"] == "wiki" and c["rag_relevante"] for c in contexts))
        self.assertEqual(
            [c["rag_url"] for c in contexts if c["top_k"] == 3 and not c["rag_relevante"]],
            irrelevant,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: FAIL (iter_rag_contexts missing).

- [ ] **Step 3: Implement iter_rag_contexts and wire run()**

Add to `main.py`:

```python
def iter_rag_contexts(perguntas, top_n_chunks_list, irrelevant_urls):
    for pair in perguntas:
        pair_wiki_url = pair.get("wiki", "")
        modes = build_rag_modes(top_n_chunks_list, pair_wiki_url, irrelevant_urls)
        for mode in modes:
            yield {
                "pair": pair,
                "top_k": mode["top_k"],
                "rag_relevante": mode["rag_relevante"],
                "rag_url": mode["rag_url"],
                "page_url": mode["page_url"],
            }
```

Then update `run(cfg)`:

```python
    irrelevant_urls = list(getattr(cfg, "WIKI_IRRELEVANT_URLS", []) or [])

    any_retrieval_requested = any(n > 0 for n in top_n_chunks_list) or len(irrelevant_urls) > 0

    ...
    if any_retrieval_requested:
        if WikiRetriever is None:
            raise RuntimeError("Retrieval solicitado, mas WikiRetriever não está disponível.")
        ...

    ...
    if any_retrieval_requested and wiki_retriever is not None:
        urls_to_validate = [p.get("wiki", "") for p in perguntas if p.get("wiki", "")]
        urls_to_validate.extend(irrelevant_urls)
        wiki_retriever.ensure_urls(urls_to_validate)

    ...
    for modelo, abordagem in cfg.MODELOS_A_AVALIAR:
        for ctx in iter_rag_contexts(perguntas, top_n_chunks_list, irrelevant_urls):
            pair = ctx["pair"]
            top_n_chunks = ctx["top_k"]
            rag_relevante = ctx["rag_relevante"]
            rag_url = ctx["rag_url"]
            page_url = ctx["page_url"]

            retriever_modo = wiki_retriever if top_n_chunks > 0 else None
            cache_extra_modo = cache_extra if top_n_chunks > 0 else None
            com_retriever = bool(retriever_modo is not None and top_n_chunks > 0)

            if cache_extra_modo:
                cache_extra_modo = f"{cache_extra_modo}|rag_relevante={int(rag_relevante)}|rag_url={rag_url}"
            elif rag_url or rag_relevante:
                cache_extra_modo = f"rag_relevante={int(rag_relevante)}|rag_url={rag_url}"

            eixo = pair["eixo"]
            for temp in cfg.TEMPERATURES:
                for rep in range(cfg.REPETICOES_POR_TEMP):
                    for tendencia_prompt, tendencia_nome in tendencias:
                        if temp == 0.0 and rep > 0:
                            continue
                        if abordagem == "gpt-sem-temperature" and temp != 0.0:
                            continue

                        tarefa_plus = obter_resposta_modelo(
                            cfg,
                            INTERVALO_SALVAMENTO,
                            cache_respostas,
                            tendencia_prompt,
                            abordagem,
                            modelo,
                            pair["p_plus"],
                            temp,
                            rep + 1,
                            top_n_chunks=top_n_chunks,
                            wiki_retriever=retriever_modo,
                            cache_extra=cache_extra_modo,
                            page_url=page_url,
                        )
                        tarefas.append({
                            "tarefa": tarefa_plus,
                            "info": {
                                "modelo": modelo,
                                "eixo": eixo,
                                "tipo_pergunta": "P+",
                                "pergunta": pair["p_plus"],
                                "temperatura": temp,
                                "repeticao": rep + 1,
                                "tendencia": tendencia_nome,
                                "pair_id": pair.get("pair_id", None),
                                "top_n_chunks": top_n_chunks,
                                "top_k": top_n_chunks,
                                "rag_relevante": rag_relevante,
                                "rag_url": rag_url,
                                "com_retriever": com_retriever,
                            },
                        })

                        tarefa_minus = obter_resposta_modelo(
                            cfg,
                            INTERVALO_SALVAMENTO,
                            cache_respostas,
                            tendencia_prompt,
                            abordagem,
                            modelo,
                            pair["p_minus"],
                            temp,
                            rep + 1,
                            top_n_chunks=top_n_chunks,
                            wiki_retriever=retriever_modo,
                            cache_extra=cache_extra_modo,
                            page_url=page_url,
                        )
                        tarefas.append({
                            "tarefa": tarefa_minus,
                            "info": {
                                "modelo": modelo,
                                "eixo": eixo,
                                "tipo_pergunta": "P-",
                                "pergunta": pair["p_minus"],
                                "temperatura": temp,
                                "repeticao": rep + 1,
                                "tendencia": tendencia_nome,
                                "pair_id": pair.get("pair_id", None),
                                "top_n_chunks": top_n_chunks,
                                "top_k": top_n_chunks,
                                "rag_relevante": rag_relevante,
                                "rag_url": rag_url,
                                "com_retriever": com_retriever,
                            },
                        })
```

Update `obter_resposta_modelo` signature and build_context call:

```python
async def obter_resposta_modelo(..., page_url: str | None = None, ...):
    ...
    if retriever is not None and top_n_chunks > 0:
        contexto = await retriever.build_context(
            afirmacao,
            top_n=top_n_chunks,
            max_chars_per_chunk=int(getattr(cfg, "WIKI_MAX_CHARS_PER_CHUNK", 900) or 900),
            page_url=page_url,
        )
```

- [ ] **Step 4: Run tests**

Run: `python -m unittest tests/test_main_helpers.py -v`

Expected: PASS (3 tests).

- [ ] **Step 5: Update config**

Edit `conf/config.yaml`:

```yaml
WIKI_IRRELEVANT_URLS:
  - "https://pt.wikipedia.org/wiki/Elevador"
  - "https://pt.wikipedia.org/wiki/Culin%C3%A1ria_da_Fran%C3%A7a"
  - "https://pt.wikipedia.org/wiki/Fotoss%C3%ADntese"
```

- [ ] **Step 6: Commit**

```bash
git add main.py conf/config.yaml tests/test_main_helpers.py
# include wiki_retrieval.py if build_context signature change was not yet committed
# git add src/main/wiki_retrieval.py

git commit -m "feat: add irrelevant RAG modes and outputs"
```

---

## Plan Self-Review

- Spec coverage: retrieval filtering, irrelevant URL runs, CSV fields, URL validation, cache isolation covered by Tasks 1-6.
- No placeholders: all steps include concrete code and commands.
- Names consistent: `build_rag_modes`, `iter_rag_contexts`, `ensure_urls`, `page_url` used consistently.
