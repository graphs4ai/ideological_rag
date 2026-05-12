import os
import sys
import tempfile
import unittest

import pandas as pd
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.analysis.processing import calcular_indice_polarizacao, carregar_e_processar_dados


class ProcessingRagColumnsTests(unittest.TestCase):
    def test_calcular_indice_polarizacao_keeps_rag_columns(self):
        df_validos = pd.DataFrame(
            {
                "modelo": ["m", "m"],
                "eixo": ["Economia", "Economia"],
                "pair_id": [1, 1],
                "tipo_pergunta": ["P+", "P-"],
                "temperatura": [0.0, 0.0],
                "tendencia": ["neutro", "neutro"],
                "pontuacao": [1, -1],
                "top_k": [3, 3],
                "rag_relevante": [True, True],
                "rag_url": ["u", "u"],
            }
        )

        _, df_ip = calcular_indice_polarizacao(df_validos)

        self.assertIn("top_k", df_ip.columns)
        self.assertIn("rag_relevante", df_ip.columns)
        self.assertIn("rag_url", df_ip.columns)

    def test_calcular_indice_polarizacao_pairs_relevant_rag_with_different_urls(self):
        df_validos = pd.DataFrame(
            {
                "modelo": ["m", "m"],
                "eixo": ["Economia", "Economia"],
                "pair_id": [1, 1],
                "tipo_pergunta": ["P+", "P-"],
                "temperatura": [0.0, 0.0],
                "tendencia": ["neutro", "neutro"],
                "pontuacao": [2, -1],
                "top_n_chunks": [3, 3],
                "top_k": [3, 3],
                "rag_relevante": [True, True],
                "rag_url": ["u-plus", "u-minus"],
                "com_retriever": [True, True],
            }
        )

        df_pares, df_ip = calcular_indice_polarizacao(df_validos)

        self.assertEqual(len(df_pares), 1)
        self.assertAlmostEqual(df_ip["indice_polarizacao"].iloc[0], 3.0)

    def test_calcular_indice_polarizacao_keeps_irrelevant_urls_separate(self):
        rows = []
        for url, plus, minus in [("u1", 2, -2), ("u2", -2, 2)]:
            rows.extend(
                [
                    {
                        "modelo": "m",
                        "eixo": "Economia",
                        "pair_id": 1,
                        "tipo_pergunta": "P+",
                        "temperatura": 0.0,
                        "tendencia": "neutro",
                        "pontuacao": plus,
                        "top_n_chunks": 3,
                        "top_k": 3,
                        "rag_relevante": False,
                        "rag_url": url,
                        "com_retriever": True,
                    },
                    {
                        "modelo": "m",
                        "eixo": "Economia",
                        "pair_id": 1,
                        "tipo_pergunta": "P-",
                        "temperatura": 0.0,
                        "tendencia": "neutro",
                        "pontuacao": minus,
                        "top_n_chunks": 3,
                        "top_k": 3,
                        "rag_relevante": False,
                        "rag_url": url,
                        "com_retriever": True,
                    },
                ]
            )
        df_validos = pd.DataFrame(rows)

        _, df_ip = calcular_indice_polarizacao(df_validos)

        self.assertEqual(set(df_ip["rag_url"]), {"u1", "u2"})
        self.assertEqual(len(df_ip), 2)

    def test_carregar_e_processar_dados_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "respostas.csv")
            df = pd.DataFrame(
                {
                    "modelo": ["m", "m", "m", "m"],
                    "eixo": ["Economia"] * 4,
                    "tipo_pergunta": ["P+", "P-", "P+", "P-"],
                    "pergunta": ["x"] * 4,
                    "temperatura": [0.0] * 4,
                    "repeticao": [1] * 4,
                    "tendencia": ["neutro"] * 4,
                    "pair_id": [1] * 4,
                    "top_n_chunks": [0, 0, 3, 3],
                    "top_k": [0, 0, 3, 3],
                    "rag_relevante": [False, False, True, True],
                    "rag_url": [None, None, "u", "u"],
                    "com_retriever": [False, False, True, True],
                    "resposta_raw": ["Concordo", "Discordo", "Concordo", "Discordo"],
                }
            )
            df.to_csv(path, index=False)

            cfg = OmegaConf.create(
                {
                    "paths": {"input_file": path},
                    "analysis": {
                        "likert_map": {
                            "Concordo fortemente": 2,
                            "Concordo": 1,
                            "Neutro": 0,
                            "Discordo": -1,
                            "Discordo fortemente": -2,
                        }
                    },
                }
            )

            df_validos = carregar_e_processar_dados(cfg)
            _, df_ip = calcular_indice_polarizacao(df_validos)

            self.assertTrue((df_ip["top_k"] == 0).any())


if __name__ == "__main__":
    unittest.main()
