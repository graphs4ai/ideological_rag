import hydra
from omegaconf import DictConfig
import os
import sys

# Garante que o python encontre o módulo src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis import processing, plotting, statistics

@hydra.main(version_base=None, config_path="conf", config_name="analysis_config")
def main(cfg: DictConfig) -> None:
    
    plotting.setup_style()

    try:
        df_validos = processing.carregar_e_processar_dados(cfg)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{cfg.paths.input_file}' não encontrado.")
        return

    df_pares, df_ip = processing.calcular_indice_polarizacao(df_validos)

    if df_ip.empty:
        print("Aviso: Nenhum dado de polarização calculado.")
        return

    # Figuras comparativas com vs sem retriever
    plotting.plot_ci_geral_por_modelo_com_vs_sem_retriever(df_ip, cfg)
    # plotting.plot_ci_por_area_com_vs_sem_retriever(df_pares, cfg)
    # plotting.plot_ipi_com_vs_sem_retriever(df_ip, cfg)
    # plotting.plot_ipi_com_vs_sem_retriever_2(df_ip, cfg)
    # plotting.plot_ipi_media_tendencias_pre_pos_retriever(df_ip, cfg)
    plotting.plot_rag_main_effect_ci(df_ip, cfg)
    plotting.plot_rag_main_effect_ci_box_plot(df_ip, cfg)
    plotting.plot_rag_main_effect_ci_box_plot_2(df_ip, cfg)
    plotting.plot_rag_main_effect_ci_box_plot_3(df_ip, cfg)
    # plotting.plot_rag_main_effect_ci_2(df_ip, cfg)
    # plotting.plot_rag_main_effect_ci_3(df_ip, cfg)
    # plotting.plot_rag_ipi_dumbbell(df_ip, cfg)
    plotting.plot_rag_ipi_dumbbell_2(df_ip, cfg)
    plotting.plot_rag_ipi_dumbbell_3(df_ip, cfg)
    # plotting.plot_rag_topic_delta_ci(df_pares, cfg)

    # Figura original. Se houver múltiplos modos, roda só no baseline (top_n_chunks==0) para não misturar.
    # if 'top_n_chunks' in df_ip.columns:
    #     df_ip_baseline = df_ip[df_ip['top_n_chunks'] == 0].copy()
    #     if not df_ip_baseline.empty:
    #         plotting.plot_user_shifts_chameleon(df_ip_baseline, cfg)
    # else:
    #     plotting.plot_user_shifts_chameleon(df_ip, cfg)

    # plotting.plot_topic_variation(df_pares, cfg)
    # plotting.plot_likert_distribution(df_validos, cfg)
    # plotting.plot_topic_dot_panel(df_pares, cfg)

    statistics.std_v(cfg)



if __name__ == "__main__":
    main()
