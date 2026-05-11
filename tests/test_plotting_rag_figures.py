import unittest

import pandas as pd

from src.analysis.plotting import _map_rag_condition


class RagConditionTests(unittest.TestCase):
    def test_map_rag_condition_baseline(self):
        row = {"top_k": 0, "rag_relevante": False}
        self.assertEqual(_map_rag_condition(row), "Baseline")

    def test_map_rag_condition_relevant(self):
        row = {"top_k": 3, "rag_relevante": True}
        self.assertEqual(_map_rag_condition(row), "Top-3 Rel")

    def test_map_rag_condition_irrelevant(self):
        row = {"top_k": 3, "rag_relevante": False}
        self.assertEqual(_map_rag_condition(row), "Top-3 Irrel")


if __name__ == "__main__":
    unittest.main()
