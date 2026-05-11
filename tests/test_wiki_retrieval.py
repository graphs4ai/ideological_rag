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

    async def test_retrieve_raises_on_missing_url(self):
        with self.assertRaises(RuntimeError):
            await self.retriever.retrieve("query", top_n=1, page_url="https://missing")

    async def test_build_context_uses_page_url(self):
        ctx = await self.retriever.build_context("query", top_n=1, page_url=self.url2)
        self.assertIn(self.url2, ctx)
        self.assertNotIn(self.url1, ctx)

    async def test_ensure_urls_accepts_source_page(self):
        os.environ["DEEPINFRA_API_KEY"] = "test"
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)

            decoded = "https://pt.wikipedia.org/wiki/Bolsa_Família"
            encoded = "https://pt.wikipedia.org/wiki/Bolsa_Fam%C3%ADlia"

            docs = [
                {
                    "doc_id": "d1",
                    "page_key": "p1",
                    "title": "Page 1",
                    "url": decoded,
                    "source_page": encoded,
                    "chunk_id": 0,
                    "text": "x",
                }
            ]
            with (store / "docs.jsonl").open("w", encoding="utf-8") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")
            np.save(store / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float32))

            retriever = WikiRetriever(store_dir=store, embedding_model="dummy")
            retriever.ensure_urls([encoded])


if __name__ == "__main__":
    unittest.main()
