import unittest

import pandas as pd

from src.analysis.plotting import (
    _compute_ci_by_condition,
    _map_rag_condition,
    _model_short_name,
    _prepare_rag_ci_by_model,
    _prepare_rag_main_effect_summary,
    _prepare_retriever_ci_comparison,
)


class RagConditionTests(unittest.TestCase):
    def test_map_rag_condition_baseline(self):
        row = {"top_k": 0, "rag_relevante": False}
        self.assertEqual(_map_rag_condition(row), "Baseline")

    def test_map_rag_condition_empty_rag_url_is_baseline(self):
        row = {"top_k": 0, "rag_relevante": False, "rag_url": ""}
        self.assertEqual(_map_rag_condition(row), "Baseline")

    def test_map_rag_condition_relevant(self):
        row = {"top_k": 3, "rag_relevante": True}
        self.assertEqual(_map_rag_condition(row), "Top-3 Rel")

    def test_map_rag_condition_relevant_without_url_after_aggregation(self):
        row = {"top_k": 3, "rag_relevante": True, "rag_url": None}
        self.assertEqual(_map_rag_condition(row), "Top-3 Rel")

    def test_map_rag_condition_irrelevant(self):
        row = {"top_k": 3, "rag_relevante": False}
        self.assertEqual(_map_rag_condition(row), "Top-3 Irrel")


class RagCiAggregationTests(unittest.TestCase):
    def test_model_short_name_removes_provider_and_verbose_suffixes(self):
        self.assertEqual(
            _model_short_name("mistralai/Mistral-Small-3.2-24B-Instruct-2506"),
            "Mistral-Small-3.2-24B",
        )
        self.assertEqual(
            _model_short_name("meta-llama/Meta-Llama-3.1-70B-Instruct"),
            "Llama-3.1-70B",
        )
        self.assertEqual(_model_short_name("sabia-3.1"), "sabia-3.1")

    def test_compute_ci_by_condition(self):
        df_ip = pd.DataFrame(
            {
                "top_k": [0, 0, 0, 3, 3, 3],
                "rag_relevante": [False, False, False, True, True, True],
                "tendencia": ["esquerda", "neutro", "direita"] * 2,
                "indice_polarizacao": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0],
            }
        )
        out = _compute_ci_by_condition(df_ip)
        baseline = out[out["condicao"] == "Baseline"]["ci"].iloc[0]
        top3 = out[out["condicao"] == "Top-3 Rel"]["ci"].iloc[0]
        self.assertAlmostEqual(baseline, 2.0)
        self.assertAlmostEqual(top3, 4.0)

    def test_prepare_rag_ci_by_model_averages_irrelevant_page_cis(self):
        df_ip = pd.DataFrame(
            {
                "modelo": ["m"] * 9,
                "top_k": [0, 0, 0, 3, 3, 3, 3, 3, 3],
                "rag_relevante": [False, False, False, False, False, False, False, False, False],
                "rag_url": [None, None, None, "u1", "u1", "u1", "u2", "u2", "u2"],
                "tendencia": ["esquerda", "neutro", "direita"] * 3,
                "indice_polarizacao": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, 2.0, 0.0, -2.0],
            }
        )

        out = _prepare_rag_ci_by_model(df_ip)

        irrelevant = out[out["condicao"] == "Top-3 Irrel"]["chameleon_index"].iloc[0]
        self.assertAlmostEqual(irrelevant, 4.0)

    def test_prepare_rag_main_effect_summary_sorts_highest_ci_first(self):
        df_ci = pd.DataFrame(
            {
                "condicao": ["Baseline", "Baseline", "Top-3 Rel", "Top-3 Rel"],
                "modelo": ["m1", "m2", "m1", "m2"],
                "chameleon_index": [5.0, 3.0, 1.0, 2.0],
            }
        )

        out = _prepare_rag_main_effect_summary(df_ci)

        self.assertEqual(out["condicao"].tolist(), ["Baseline", "Top-3 Rel"])

    def test_prepare_retriever_ci_comparison_uses_only_relevant_rag(self):
        df_ip = pd.DataFrame(
            {
                "modelo": ["m"] * 9,
                "top_n_chunks": [0, 0, 0, 3, 3, 3, 3, 3, 3],
                "top_k": [0, 0, 0, 3, 3, 3, 3, 3, 3],
                "rag_relevante": [False, False, False, True, True, True, False, False, False],
                "rag_url": [None, None, None, "u", "u", "u", "x", "x", "x"],
                "tendencia": ["esquerda", "neutro", "direita"] * 3,
                "indice_polarizacao": [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -4.0, 0.0, 4.0],
            }
        )

        out = _prepare_retriever_ci_comparison(df_ip)

        self.assertEqual(set(out["retriever"]), {"Without retriever", "With relevant retriever"})
        with_retriever = out[out["retriever"] == "With relevant retriever"]["chameleon_index"].iloc[0]
        self.assertAlmostEqual(with_retriever, 4.0)


if __name__ == "__main__":
    unittest.main()
