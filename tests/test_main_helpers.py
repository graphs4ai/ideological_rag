import json
import hashlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np

from main import build_rag_modes, iter_rag_contexts, obter_resposta_modelo
from src.main.utils import gerar_chave_cache
from src.main.wiki_retrieval import WikiRetriever


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


class ResponseCacheRagTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_cache_key_uses_retrieved_context_not_store_fingerprint(self):
        context = "\nPágina: Page\nURL: u\n\n- cached chunk"
        base_extra = "wiki_store=wiki_faiss_store|wiki_emb=dummy|rag_relevante=1|rag_url=u"
        context_hash = hashlib.md5(context.encode("utf-8")).hexdigest()
        cache_extra = f"{base_extra}|wiki_topn=3|wiki_context={context_hash}"
        chave = gerar_chave_cache(
            "model",
            "afirmacao",
            0.0,
            1,
            "prompt",
            extra=cache_extra,
        )
        cache = {chave: "Concordo"}

        class FakeRetriever:
            def __init__(self):
                self.calls = 0

            async def build_context(self, query, *, top_n, max_chars_per_chunk, page_url):
                self.calls += 1
                self.args = (query, top_n, max_chars_per_chunk, page_url)
                return context

        retriever = FakeRetriever()
        cfg = SimpleNamespace(
            TOP_N_CHUNKS=[0, 3],
            WIKI_MAX_CHARS_PER_CHUNK=900,
            ARQUIVO_CACHE="unused.pkl",
        )

        with patch("main.chamar_api_provider", new=AsyncMock(side_effect=AssertionError("cache miss"))):
            resposta = await obter_resposta_modelo(
                cfg,
                999999,
                cache,
                "prompt",
                "deepinfra",
                "model",
                "afirmacao",
                0.0,
                1,
                top_n_chunks=3,
                page_url="u",
                wiki_retriever=retriever,
                cache_extra=base_extra,
            )

        self.assertEqual(resposta, "Concordo")
        self.assertEqual(retriever.calls, 1)


if __name__ == "__main__":
    unittest.main()
