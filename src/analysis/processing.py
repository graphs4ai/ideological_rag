import pandas as pd
from omegaconf import DictConfig

def carregar_e_processar_dados(cfg: DictConfig) -> pd.DataFrame:
    df_resultados = pd.read_csv(cfg.paths.input_file)
    likert_map = dict(cfg.analysis.likert_map)

    if 'top_n_chunks' not in df_resultados.columns:
        if 'com_retriever' in df_resultados.columns:
            df_resultados['top_n_chunks'] = df_resultados['com_retriever'].fillna(False).astype(bool).map(lambda x: 1 if x else 0)
        else:
            df_resultados['top_n_chunks'] = 0

    if 'com_retriever' not in df_resultados.columns:
        df_resultados['com_retriever'] = df_resultados['top_n_chunks'].fillna(0).astype(int) > 0

    df_resultados['top_n_chunks'] = df_resultados['top_n_chunks'].fillna(0).astype(int)
    df_resultados['com_retriever'] = df_resultados['com_retriever'].fillna(False).astype(bool)
    
    df_resultados['pontuacao'] = df_resultados['resposta_raw'].map(likert_map)
    df_validos = df_resultados.dropna(subset=['pontuacao']).copy()
    df_validos['pontuacao'] = df_validos['pontuacao'].astype(int)
    return df_validos

def carregar_e_processar_ambos(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both original and negated CSVs for comparison figures."""
    likert_map = dict(cfg.analysis.likert_map)

    df_orig = pd.read_csv("dados/respostas.csv")
    df_orig['pontuacao'] = df_orig['resposta_raw'].map(likert_map)
    df_orig = df_orig.dropna(subset=['pontuacao']).copy()
    df_orig['pontuacao'] = df_orig['pontuacao'].astype(int)

    df_neg = pd.read_csv("dados/respostas.csv")
    df_neg['pontuacao'] = df_neg['resposta_raw'].map(likert_map)
    df_neg = df_neg.dropna(subset=['pontuacao']).copy()
    df_neg['pontuacao'] = df_neg['pontuacao'].astype(int)

    return df_orig, df_neg

def calcular_indice_polarizacao(df_validos: pd.DataFrame):
    df_validos = df_validos.copy()

    if {'rag_url', 'rag_relevante', 'top_k'}.issubset(df_validos.columns):
        rag_url_not_empty = df_validos['rag_url'].notna() & (df_validos['rag_url'].astype(str).str.strip() != '')
        is_irrelevant_control = (
            df_validos['top_k'].fillna(0).astype(int).gt(0)
            & ~df_validos['rag_relevante'].fillna(False).astype(bool)
            & rag_url_not_empty
        )
        df_validos['rag_context_group'] = df_validos['rag_url'].where(is_irrelevant_control)

    # Agrupa médias
    retrieval_cols: list[str] = []
    if 'top_n_chunks' in df_validos.columns:
        retrieval_cols.append('top_n_chunks')
    if 'com_retriever' in df_validos.columns:
        retrieval_cols.append('com_retriever')
    if 'top_k' in df_validos.columns:
        retrieval_cols.append('top_k')
    if 'rag_relevante' in df_validos.columns:
        retrieval_cols.append('rag_relevante')
    if 'rag_context_group' in df_validos.columns:
        retrieval_cols.append('rag_context_group')

    cols_group = ['modelo', 'eixo', 'pair_id', 'tipo_pergunta', 'temperatura', 'tendencia'] + retrieval_cols
    df_medias = df_validos.groupby(cols_group, dropna=False)['pontuacao'].mean().reset_index()

    # Separa P+ e P-
    df_p_plus = df_medias[df_medias['tipo_pergunta'] == 'P+'].rename(columns={'pontuacao': 'media_R_plus'})
    df_p_minus = df_medias[df_medias['tipo_pergunta'] == 'P-'].rename(columns={'pontuacao': 'media_R_minus'})

    # Merge
    merge_keys = ['modelo', 'eixo', 'pair_id', 'temperatura', 'tendencia'] + retrieval_cols

    df_pares = pd.merge(df_p_plus, df_p_minus, on=merge_keys, how='inner')
    
    df_pares['diferenca_R'] = df_pares['media_R_plus'] - df_pares['media_R_minus']
    
    # IP Médio
    df_ip = df_pares.groupby(['modelo', 'temperatura', 'tendencia'] + retrieval_cols, dropna=False)['diferenca_R'].mean().reset_index()
    df_ip = df_ip.rename(columns={'diferenca_R': 'indice_polarizacao'})

    if 'rag_context_group' in df_pares.columns:
        df_pares['rag_url'] = df_pares['rag_context_group']
    if 'rag_context_group' in df_ip.columns:
        df_ip['rag_url'] = df_ip['rag_context_group']
    
    return df_pares, df_ip

def get_model_size(model_name: str, params_db: dict) -> float | None:
    model_name_lower = str(model_name).lower()
    return params_db.get(model_name_lower)
