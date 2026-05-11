from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Dependência ausente: numpy. Instale para usar retrieval do wiki_faiss_store."
    ) from e

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # lazy error

from openai import AsyncOpenAI


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


@dataclass(frozen=True)
class WikiChunk:
    rank: int
    score: float
    doc_id: str
    page_key: str
    title: str
    url: str
    chunk_id: int
    text: str


@dataclass
class WikiFaissStore:
    docs: list[dict[str, Any]]
    embeddings: np.ndarray
    index: Any  # faiss.Index
    fingerprint: str | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def load_store(store_dir: str | Path = "wiki_faiss_store") -> WikiFaissStore:
    if faiss is None:  # pragma: no cover
        raise ImportError(
            "Dependência ausente: faiss. Instale `faiss-cpu` para usar retrieval."
        )

    store_dir = Path(store_dir)
    docs_path = store_dir / "docs.jsonl"
    embeddings_path = store_dir / "embeddings.npy"
    faiss_path = store_dir / "index.faiss"
    manifest_path = store_dir / "manifest.json"

    if not docs_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Store incompleto em '{store_dir}'. Esperado: docs.jsonl e embeddings.npy"
        )

    docs = _read_jsonl(docs_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    if len(docs) != int(embeddings.shape[0]):
        raise RuntimeError(
            f"Inconsistência no índice: {len(docs)} docs, mas {embeddings.shape[0]} embeddings."
        )

    if faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
    else:
        # Fallback: reconstrói o índice.
        if embeddings.size == 0 or embeddings.shape[0] == 0:
            raise RuntimeError("Embeddings vazios; não é possível construir índice FAISS.")
        dim = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

    fingerprint: str | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = str(manifest.get("updated_at") or "")
        except Exception:
            fingerprint = None

    return WikiFaissStore(docs=docs, embeddings=embeddings, index=index, fingerprint=fingerprint)


class WikiRetriever:
    def __init__(
        self,
        *,
        store_dir: str | Path = "wiki_faiss_store",
        embedding_model: str = "google/embeddinggemma-300m",
        deepinfra_api_key: str | None = None,
        deepinfra_base_url: str = "https://api.deepinfra.com/v1/openai",
        max_concurrent_embeddings: int = 5,
        logger: logging.Logger | None = None,
    ):
        self._logger = logger or logging.getLogger(__name__)
        self._store = load_store(store_dir)
        self.embedding_model = embedding_model

        self._url_to_indices: dict[str, np.ndarray] = {}
        for idx, doc in enumerate(self._store.docs):
            url = str(doc.get("url", ""))
            source_page = str(doc.get("source_page", ""))
            candidates = [url, source_page]
            if source_page:
                candidates.append(unquote(source_page))
            for candidate in candidates:
                if not candidate:
                    continue
                self._url_to_indices.setdefault(candidate, []).append(idx)
        self._url_to_indices = {
            url: np.array(indices, dtype=np.int64)
            for url, indices in self._url_to_indices.items()
        }

        api_key = deepinfra_api_key or os.getenv("DEEPINFRA_API_KEY") or os.getenv("DEEPINFRA_TOKEN")
        if not api_key:
            raise RuntimeError(
                "Defina DEEPINFRA_API_KEY (ou DEEPINFRA_TOKEN) no ambiente para gerar embeddings do query."
            )

        self._client = AsyncOpenAI(api_key=api_key, base_url=deepinfra_base_url)
        self._embed_semaphore = asyncio.Semaphore(int(max_concurrent_embeddings))

        self._result_cache: dict[tuple[str, int, str], asyncio.Future[list[WikiChunk]]] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def fingerprint(self) -> str | None:
        return self._store.fingerprint

    def has_url(self, url: str) -> bool:
        return url in self._url_to_indices

    def ensure_urls(self, urls: list[str]) -> None:
        missing = [u for u in urls if u not in self._url_to_indices]
        if missing:
            raise RuntimeError(f"URL(s) não encontradas no índice: {missing}")

    async def _embed_query(self, text: str) -> np.ndarray:
        async with self._embed_semaphore:
            try:
                resp = await self._client.embeddings.create(
                    model=self.embedding_model,
                    input=[text],
                    encoding_format="float",
                )
            except TypeError:
                # Compatibilidade com versões/servidores que não aceitam encoding_format.
                resp = await self._client.embeddings.create(
                    model=self.embedding_model,
                    input=[text],
                )

        emb = resp.data[0].embedding
        query_emb = np.array([emb], dtype=np.float32)
        return l2_normalize(query_emb).astype(np.float32)

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

        results: list[WikiChunk] = []
        for rank, global_idx in enumerate(idxs[0], start=1):
            if int(global_idx) < 0:
                continue
            doc = self._store.docs[int(global_idx)]
            results.append(
                WikiChunk(
                    rank=rank,
                    score=float(scores[0][rank - 1]),
                    doc_id=str(doc.get("doc_id", "")),
                    page_key=str(doc.get("page_key", "")),
                    title=str(doc.get("title", "")),
                    url=str(doc.get("url", "")),
                    chunk_id=int(doc.get("chunk_id", -1)),
                    text=str(doc.get("text", "")),
                )
            )

        return results

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