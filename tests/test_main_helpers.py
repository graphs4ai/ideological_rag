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
