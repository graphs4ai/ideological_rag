import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import build_rag_modes, iter_rag_contexts
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


if __name__ == "__main__":
    unittest.main()
