import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from omegaconf import DictConfig

def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0})

def save_fig(fig, name: str, cfg: DictConfig):
    if cfg.analysis.save_plots:
        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "svg"), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(cfg.paths.output_dir, "pdf"), exist_ok=True)
        path_svg = os.path.join(cfg.paths.output_dir, f"svg/{name}.svg")
        path_png = os.path.join(cfg.paths.output_dir, f"png/{name}.png")
        path_pdf = os.path.join(cfg.paths.output_dir, f"pdf/{name}.pdf")
        fig.savefig(path_svg, format="svg", bbox_inches='tight')
        fig.savefig(path_png, format="png", bbox_inches='tight')
        fig.savefig(path_pdf, format="pdf", bbox_inches='tight')
        print(f"Figuras salvas em: {path_png}")

def plot_user_shifts_chameleon(df_ip: pd.DataFrame, cfg: DictConfig):
    """
    User-Conditioned Shifts & Chameleon Index.
    """
    # Prepare data - calculate mean and std IP per model and condition
    df_shifts = df_ip.groupby(['modelo', 'tendencia'])['indice_polarizacao'].agg(['mean', 'std']).reset_index()
    df_shifts.columns = ['modelo', 'tendencia', 'ip_mean', 'ip_std']
        
    # Pivot mean values
    df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_mean')
    df_pivot = df_pivot.reset_index()
    
    # Pivot std values
    df_pivot_std = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_std')
    df_pivot_std = df_pivot_std.reset_index()
    df_pivot_std.columns = ['modelo', 'esquerda_std', 'neutro_std', 'direita_std']
    
    # Merge mean and std
    df_pivot = df_pivot.merge(df_pivot_std, on='modelo')
    
    # Calculate Chameleon Index (sum of absolute shifts from neutral)
    df_pivot['shift_left'] = abs(df_pivot['esquerda'] - df_pivot['neutro'])
    df_pivot['shift_right'] = abs(df_pivot['direita'] - df_pivot['neutro'])
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']
    
    # Sort by chameleon index for Panel B
    df_pivot_sorted = df_pivot.sort_values('chameleon_index', ascending=False)
    
    df_pivot_a = df_pivot.sort_values('neutro')
    y_pos = np.arange(len(df_pivot_a))
    colors_gradient = plt.cm.YlOrRd(df_pivot_sorted['chameleon_index'] / df_pivot_sorted['chameleon_index'].max())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 14), gridspec_kw={'width_ratios': [2, 1]})
    # Panel A
    for i, row in df_pivot_a.iterrows():
        x_values = [row['esquerda'], row['neutro'], row['direita']]
        ax1.plot(x_values, [y_pos[df_pivot_a.index.get_loc(i)]] * 3, color='gray', alpha=0.3, linewidth=1)
    ax1.errorbar(df_pivot_a['esquerda'], y_pos, xerr=df_pivot_a['esquerda_std'],
                fmt='o', markersize=15, color='#e74c3c', ecolor='#e74c3c',
                alpha=0.8, label='Left-Wing User', zorder=3, capsize=3, capthick=1.5)
    ax1.errorbar(df_pivot_a['neutro'], y_pos, xerr=df_pivot_a['neutro_std'],
                fmt='o', markersize=15, color='#95a5a6', ecolor='#95a5a6',
                alpha=0.8, label='No-Context User', zorder=3, capsize=3, capthick=1.5)
    ax1.errorbar(df_pivot_a['direita'], y_pos, xerr=df_pivot_a['direita_std'],
                fmt='o', markersize=15, color='#3498db', ecolor='#3498db',
                alpha=0.8, label='Right-Wing User', zorder=3, capsize=3, capthick=1.5)
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_pivot_a['modelo'], fontsize=25)
    ax1.set_xlabel('Ideological Position Index (IPI)', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Model', fontsize=26, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=21)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=26, ncol=3, frameon=False)
    ax1.grid(axis='x', alpha=0.3, linestyle=':')
    ax1.set_xlim(-4, 4)
    ax1.text(
        -0.40, 1.01, 'A',
        transform=ax1.transAxes,
        fontsize=28,
        fontweight='bold',
        va='top',
        ha='right'
    )
    ax2.text(
        -0.40, 1.01, 'B',
        transform=ax2.transAxes,
        fontsize=28,
        fontweight='bold',
        va='top',
        ha='right'
    )
    y_pos_b = np.arange(len(df_pivot_sorted))
    ax2.barh(y_pos_b, df_pivot_sorted['chameleon_index'], color=colors_gradient, alpha=0.85, height=0.7)
    ax2.set_yticks(y_pos_b)
    ax2.set_yticklabels(df_pivot_sorted['modelo'], fontsize=25)
    ax2.set_xlabel('Chameleon Index (CI)', fontsize=26, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=21)
    ax2.grid(axis='x', alpha=0.3, linestyle=':')
    ax2.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, "user_shifts_chameleon", cfg)
    plt.close()


def _pick_retrieval_modes(df: pd.DataFrame) -> tuple[int, int]:
    """Pick (no_retrieval, with_retrieval) top_n values from data.

    - Prefere 0 como baseline quando existir.
    - Para 'com retrieval', pega o maior top_n > 0.
    - Se só existir um modo, retorna (modo, modo).
    """
    if 'top_n_chunks' not in df.columns or df['top_n_chunks'].dropna().empty:
        return 0, 0

    vals = sorted(set(int(x) for x in df['top_n_chunks'].dropna().tolist()))
    if not vals:
        return 0, 0

    no_mode = 0 if 0 in vals else vals[0]
    positives = [v for v in vals if v > 0]
    with_mode = max(positives) if positives else no_mode
    return no_mode, with_mode


_RAG_CONDITION_ORDER = [
    "Baseline",
    "Top-1 Rel",
    "Top-3 Rel",
    "Top-5 Rel",
    "Top-3 Irrel",
]


def _map_rag_condition(row: dict) -> str | None:
    rag_url = row.get("rag_url")
    if rag_url is not None and (pd.isna(rag_url) or str(rag_url).strip() == ""):
        return "Baseline"

    top_k = int(row.get("top_k", 0) or 0)
    rag_relevante = bool(row.get("rag_relevante", False))
    if top_k == 0:
        return "Baseline"
    if top_k == 1 and rag_relevante:
        return "Top-1 Rel"
    if top_k == 3 and rag_relevante:
        return "Top-3 Rel"
    if top_k == 5 and rag_relevante:
        return "Top-5 Rel"
    if top_k == 3 and not rag_relevante:
        return "Top-3 Irrel"
    return None


def _compute_ci_by_condition(df_ip: pd.DataFrame) -> pd.DataFrame:
    df_use = df_ip.copy()
    df_use["condicao"] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use["condicao"].notna()]

    df_shifts = (
        df_use.groupby(["condicao", "tendencia"])["indice_polarizacao"]
        .mean()
        .reset_index()
    )
    df_pivot = df_shifts.pivot_table(index="condicao", columns="tendencia", values="indice_polarizacao")
    df_pivot = df_pivot.reset_index()

    if not {"esquerda", "neutro", "direita"}.issubset(set(df_pivot.columns)):
        return pd.DataFrame(columns=["condicao", "ci"])

    df_pivot["shift_left"] = (df_pivot["esquerda"] - df_pivot["neutro"]).abs()
    df_pivot["shift_right"] = (df_pivot["direita"] - df_pivot["neutro"]).abs()
    df_pivot["ci"] = df_pivot["shift_left"] + df_pivot["shift_right"]

    return df_pivot[["condicao", "ci"]]


def _compute_ci_from_ip(df_ip: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute Chameleon Index from df_ip aggregated per tendency."""
    df_shifts = df_ip.groupby(group_cols + ['tendencia'])['indice_polarizacao'].agg(['mean', 'std']).reset_index()
    df_shifts = df_shifts.rename(columns={'mean': 'ip_mean', 'std': 'ip_std'})

    df_pivot = df_shifts.pivot_table(index=group_cols, columns='tendencia', values='ip_mean').reset_index()
    if not {'esquerda', 'neutro', 'direita'}.issubset(set(df_pivot.columns)):
        return pd.DataFrame(columns=group_cols + ['chameleon_index'])

    df_pivot['shift_left'] = (df_pivot['esquerda'] - df_pivot['neutro']).abs()
    df_pivot['shift_right'] = (df_pivot['direita'] - df_pivot['neutro']).abs()
    df_pivot['chameleon_index'] = df_pivot['shift_left'] + df_pivot['shift_right']
    return df_pivot


def _prepare_retriever_ci_comparison(df_ip: pd.DataFrame) -> pd.DataFrame:
    df_use = df_ip.copy()
    df_use['condicao'] = df_use.apply(_map_rag_condition, axis=1)

    df_without = df_use[df_use['condicao'] == 'Baseline'].copy()
    df_with = df_use[df_use['condicao'].isin(['Top-1 Rel', 'Top-3 Rel', 'Top-5 Rel'])].copy()
    if not df_with.empty:
        mode_col = 'top_n_chunks' if 'top_n_chunks' in df_with.columns else 'top_k'
        with_mode = int(df_with[mode_col].dropna().astype(int).max())
        df_with = df_with[df_with[mode_col].fillna(0).astype(int) == with_mode].copy()

    df_without['retriever'] = 'Without retriever'
    df_with['retriever'] = 'With relevant retriever'
    df_compare = pd.concat([df_without, df_with], ignore_index=True)
    if df_compare.empty:
        return pd.DataFrame(columns=['modelo', 'retriever', 'chameleon_index'])

    return _compute_ci_from_ip(df_compare, group_cols=['modelo', 'retriever'])


def plot_ci_geral_por_modelo_com_vs_sem_retriever(df_ip: pd.DataFrame, cfg: DictConfig):
    """CI geral por modelo, comparando com vs sem retriever."""
    if df_ip.empty:
        print("Aviso: df_ip vazio; pulando figura CI com vs sem retriever.")
        return

    df_ci = _prepare_retriever_ci_comparison(df_ip)
    if df_ci.empty:
        print("Aviso: não foi possível calcular CI (faltam tendências esquerda/neutro/direita).")
        return

    # Ordena modelos pela média de CI (desc)
    order = (
        df_ci.groupby('modelo')['chameleon_index']
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(18, max(6, 0.6 * len(order))))
    sns.barplot(
        data=df_ci,
        y='modelo',
        x='chameleon_index',
        hue='retriever',
        order=order,
        ax=ax,
        orient='h',
        dodge=True,
    )
    ax.set_xlabel('Chameleon Index (CI)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Model', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.legend(title=None, fontsize=12, frameon=False)
    plt.tight_layout()
    save_fig(fig, 'ci_geral_por_modelo_com_vs_sem_retriever', cfg)
    plt.close()


def plot_ci_por_area_com_vs_sem_retriever(df_pares: pd.DataFrame, cfg: DictConfig):
    """CI por área (eixo), comparando com vs sem retriever.

    Saída: dois heatmaps lado a lado (sem vs com), no estilo da figura de variação por tópico.
    """
    if df_pares.empty or 'top_n_chunks' not in df_pares.columns:
        print("Aviso: df_pares vazio ou sem coluna top_n_chunks; pulando figura CI por área.")
        return

    no_mode, with_mode = _pick_retrieval_modes(df_pares)
    modes = [no_mode, with_mode]
    labels = {no_mode: 'Sem retriever', with_mode: 'Com retriever'}

    topic_translation = {
        'Políticas Sociais': 'Welfare',
        'Segurança Pública': 'Security',
        'Economia': 'Economy',
        'Meio Ambiente': 'Environment',
        'Educação e Cultura': 'Education and Culture',
        'Corrupção e Justiça': 'Corruption and Justice',
        'Instituições Democráticas': 'Democratic Institutions',
    }

    fig, axes = plt.subplots(1, 2, figsize=(30, 14), sharey=True)

    heatmaps: dict[int, pd.DataFrame] = {}
    for mode in modes:
        df_mode = df_pares[df_pares['top_n_chunks'] == mode].copy()
        if df_mode.empty:
            heatmaps[mode] = pd.DataFrame()
            continue

        df_topic = df_mode.groupby(['modelo', 'eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
        df_topic_pivot = df_topic.pivot_table(index=['modelo', 'eixo'], columns='tendencia', values='diferenca_R').reset_index()

        if not {'esquerda', 'neutro', 'direita'}.issubset(set(df_topic_pivot.columns)):
            heatmaps[mode] = pd.DataFrame()
            continue

        df_topic_pivot['shift_left'] = (df_topic_pivot['esquerda'] - df_topic_pivot['neutro']).abs()
        df_topic_pivot['shift_right'] = (df_topic_pivot['direita'] - df_topic_pivot['neutro']).abs()
        df_topic_pivot['chameleon_index'] = df_topic_pivot['shift_left'] + df_topic_pivot['shift_right']

        hm = df_topic_pivot.pivot(index='modelo', columns='eixo', values='chameleon_index')
        hm.columns = [topic_translation.get(col, col) for col in hm.columns]
        heatmaps[mode] = hm

    if heatmaps[no_mode].empty and heatmaps[with_mode].empty:
        print("Aviso: não foi possível calcular CI por área para nenhum modo.")
        plt.close(fig)
        return

    # Alinha índice/colunas para facilitar comparação visual
    all_models = sorted(set(heatmaps[no_mode].index.tolist() + heatmaps[with_mode].index.tolist()))
    all_topics = sorted(set(heatmaps[no_mode].columns.tolist() + heatmaps[with_mode].columns.tolist()))
    for mode in modes:
        heatmaps[mode] = heatmaps[mode].reindex(index=all_models, columns=all_topics)

    vmax = max(
        float(heatmaps[no_mode].max().max()) if not heatmaps[no_mode].empty else 0,
        float(heatmaps[with_mode].max().max()) if not heatmaps[with_mode].empty else 0,
    )
    vmax = max(vmax, 1e-6)

    for ax, mode in zip(axes, modes):
        sns.heatmap(
            heatmaps[mode],
            annot=False,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Chameleon Index (CI)'},
            linewidths=0.5,
            linecolor='white',
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(labels[mode], fontsize=22, fontweight='bold')
        ax.set_xlabel('Topic', fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    axes[0].set_ylabel('Model', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'ci_por_area_com_vs_sem_retriever', cfg)
    plt.close()


def plot_ipi_com_vs_sem_retriever(df_ip: pd.DataFrame, cfg: DictConfig):
    """Comparação do IPI com vs sem retriever."""
    if df_ip.empty or 'top_n_chunks' not in df_ip.columns:
        print("Aviso: df_ip vazio ou sem coluna top_n_chunks; pulando figura IPI com vs sem retriever.")
        return

    no_mode, with_mode = _pick_retrieval_modes(df_ip)
    modes = [no_mode, with_mode]
    titles = {no_mode: 'Sem retriever', with_mode: 'Com retriever'}

    fig, axes = plt.subplots(1, 2, figsize=(32, 14), sharey=True)

    for ax, mode in zip(axes, modes):
        df_mode = df_ip[df_ip['top_n_chunks'] == mode].copy()
        if df_mode.empty:
            ax.set_title(f"{titles[mode]} (sem dados)")
            ax.axis('off')
            continue

        df_shifts = df_mode.groupby(['modelo', 'tendencia'])['indice_polarizacao'].agg(['mean', 'std']).reset_index()
        df_shifts.columns = ['modelo', 'tendencia', 'ip_mean', 'ip_std']

        df_pivot = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_mean').reset_index()
        df_pivot_std = df_shifts.pivot(index='modelo', columns='tendencia', values='ip_std').reset_index()
        if not {'esquerda', 'neutro', 'direita'}.issubset(set(df_pivot.columns)):
            ax.set_title(f"{titles[mode]} (faltam tendências)")
            ax.axis('off')
            continue

        # Garantir colunas std (caso alguma tendência esteja ausente)
        df_pivot_std = df_pivot_std.rename(columns=lambda c: f"{c}_std" if c in ['esquerda', 'neutro', 'direita'] else c)
        df_plot = df_pivot.merge(df_pivot_std, on='modelo', how='left')

        df_plot = df_plot.sort_values('neutro')
        y_pos = np.arange(len(df_plot))

        for i, row in df_plot.iterrows():
            x_values = [row['esquerda'], row['neutro'], row['direita']]
            ax.plot(x_values, [y_pos[df_plot.index.get_loc(i)]] * 3, color='gray', alpha=0.3, linewidth=1)

        ax.errorbar(
            df_plot['esquerda'],
            y_pos,
            xerr=df_plot.get('esquerda_std', pd.Series([0] * len(df_plot))),
            fmt='o',
            markersize=12,
            color='#e74c3c',
            ecolor='#e74c3c',
            alpha=0.85,
            label='Left-Wing User',
            zorder=3,
            capsize=3,
            capthick=1.2,
        )
        ax.errorbar(
            df_plot['neutro'],
            y_pos,
            xerr=df_plot.get('neutro_std', pd.Series([0] * len(df_plot))),
            fmt='o',
            markersize=12,
            color='#95a5a6',
            ecolor='#95a5a6',
            alpha=0.85,
            label='No-Context User',
            zorder=3,
            capsize=3,
            capthick=1.2,
        )
        ax.errorbar(
            df_plot['direita'],
            y_pos,
            xerr=df_plot.get('direita_std', pd.Series([0] * len(df_plot))),
            fmt='o',
            markersize=12,
            color='#3498db',
            ecolor='#3498db',
            alpha=0.85,
            label='Right-Wing User',
            zorder=3,
            capsize=3,
            capthick=1.2,
        )

        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['modelo'], fontsize=13)
        ax.set_xlabel('Ideological Position Index (IPI)', fontsize=18, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.set_xlim(-4, 4)
        ax.set_title(titles[mode], fontsize=22, fontweight='bold')

    axes[0].set_ylabel('Model', fontsize=18, fontweight='bold')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=16)
    plt.tight_layout()
    save_fig(fig, 'ipi_com_vs_sem_retriever', cfg)
    plt.close()


def plot_ipi_com_vs_sem_retriever_2(df_ip: pd.DataFrame, cfg: DictConfig):
    """Figura aprimorada: comparação modelo-a-modelo (sem vs com retriever).

    Para cada modelo, desenha duas linhas (sem/com) uma abaixo da outra,
    cada uma conectando esquerda-neutro-direita. Isso facilita comparar o efeito
    do retriever dentro de cada modelo.
    """
    if df_ip.empty or 'top_n_chunks' not in df_ip.columns:
        print("Aviso: df_ip vazio ou sem coluna top_n_chunks; pulando ipi_com_vs_sem_retriever_2.")
        return

    no_mode, with_mode = _pick_retrieval_modes(df_ip)
    modes = [no_mode, with_mode]
    mode_label = {no_mode: f'sem (top_n={no_mode})', with_mode: f'com (top_n={with_mode})'}

    # Agrega uma vez para obter médias e desvios por modo
    df_use = df_ip[df_ip['top_n_chunks'].isin(modes)].copy()
    if df_use.empty:
        print("Aviso: nenhum dado encontrado para os modos de retrieval selecionados.")
        return

    df_shifts_all = df_use.groupby(['top_n_chunks', 'modelo', 'tendencia'])['indice_polarizacao'].agg(['mean', 'std']).reset_index()
    df_shifts_all.columns = ['top_n_chunks', 'modelo', 'tendencia', 'ip_mean', 'ip_std']

    # Define uma ordem de modelos única (baseada no neutro do modo sem retriever quando existir)
    baseline_neutral = df_shifts_all[(df_shifts_all['top_n_chunks'] == no_mode) & (df_shifts_all['tendencia'] == 'neutro')][
        ['modelo', 'ip_mean']
    ].dropna()

    if not baseline_neutral.empty:
        model_order = baseline_neutral.sort_values('ip_mean')['modelo'].tolist()
    else:
        # fallback: neutro agregado em qualquer modo
        neutral_any = df_shifts_all[df_shifts_all['tendencia'] == 'neutro'][['modelo', 'ip_mean']].dropna()
        model_order = neutral_any.groupby('modelo')['ip_mean'].mean().sort_values().index.tolist()

    if not model_order:
        model_order = sorted(df_shifts_all['modelo'].unique().tolist())

    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    tend_order = ['esquerda', 'neutro', 'direita']
    tend_labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}

    # Pivôs por modo para facilitar o plot
    piv_mean = {}
    piv_std = {}
    for mode in modes:
        df_mode = df_shifts_all[df_shifts_all['top_n_chunks'] == mode].copy()
        piv_mean[mode] = df_mode.pivot(index='modelo', columns='tendencia', values='ip_mean').reindex(model_order)
        piv_std[mode] = df_mode.pivot(index='modelo', columns='tendencia', values='ip_std').reindex(model_order)

    # Layout: duas linhas por modelo
    y_positions = []
    y_labels = []
    for i, model in enumerate(model_order):
        y_positions.extend([2 * i, 2 * i + 1])
        y_labels.extend([f"{model} — {mode_label[no_mode]}", f"{model} — {mode_label[with_mode]}"])

    fig, ax = plt.subplots(figsize=(28, max(12, 0.55 * len(y_labels))))

    # Conectores e pontos por modo
    for i, model in enumerate(model_order):
        for j, mode in enumerate(modes):
            y = 2 * i + j
            row = piv_mean[mode].loc[model] if model in piv_mean[mode].index else None
            if row is None or row.isna().any():
                continue

            # Conector (esquerda -> neutro -> direita)
            ax.plot(
                [row['esquerda'], row['neutro'], row['direita']],
                [y, y, y],
                color='gray',
                alpha=0.25 if mode == no_mode else 0.40,
                linewidth=1.2 if mode == no_mode else 1.6,
            )

    # Pontos com cores por tendência e estilo por modo (sem: círculo vazio / com: círculo cheio)
    for tend in tend_order:
        for j, mode in enumerate(modes):
            xs = []
            ys = []
            xerrs = []
            for i, model in enumerate(model_order):
                y = 2 * i + j
                if model not in piv_mean[mode].index:
                    continue
                val = piv_mean[mode].loc[model].get(tend, np.nan)
                if pd.isna(val):
                    continue
                xs.append(val)
                ys.append(y)
                xerrs.append(float(piv_std[mode].loc[model].get(tend, 0) or 0))

            if not xs:
                continue

            # sem: facecolor none (vazio); com: facecolor sólido
            mfc = 'none' if mode == no_mode else colors[tend]
            mec = colors[tend]
            label = tend_labels[tend] if (mode == with_mode) else None  # legenda só uma vez
            ax.errorbar(
                xs,
                ys,
                xerr=xerrs,
                fmt='o',
                markersize=8,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                markeredgewidth=1.8,
                color=mec,
                ecolor=mec,
                alpha=0.9,
                capsize=2.5,
                linewidth=0,
                zorder=3,
                label=label,
            )

    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel('Ideological Position Index (IPI)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Model (modo)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_xlim(-4, 4)
    ax.set_title('IPI por modelo: comparação sem vs com retriever', fontsize=18, fontweight='bold')

    # Legenda (cores = tendência; estilo = modo descrito nos rótulos do eixo Y)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=13)

    plt.tight_layout()
    save_fig(fig, 'ipi_com_vs_sem_retriever_2', cfg)
    plt.close()


def plot_ipi_media_tendencias_pre_pos_retriever(df_ip: pd.DataFrame, cfg: DictConfig):
    """Figura: média do IPI (left/no-context/right) antes vs depois do retriever.

    - Agrega por tendência, comparando top_n_chunks=0 (sem) vs top_n_chunks>0 (com).
    - Plot em colunas (barras agrupadas).
    """
    if df_ip.empty or 'top_n_chunks' not in df_ip.columns:
        print("Aviso: df_ip vazio ou sem coluna top_n_chunks; pulando ipi_media_tendencias_pre_pos_retriever.")
        return

    no_mode, with_mode = _pick_retrieval_modes(df_ip)
    df_use = df_ip[df_ip['top_n_chunks'].isin([no_mode, with_mode])].copy()
    if df_use.empty:
        print("Aviso: nenhum dado encontrado para os modos de retrieval selecionados.")
        return

    # Média do IPI por tendência (agregando modelos/temperaturas)
    df_mean = (
        df_use.groupby(['top_n_chunks', 'tendencia'])['indice_polarizacao']
        .mean()
        .reset_index()
        .rename(columns={'indice_polarizacao': 'ipi_mean'})
    )
    df_mean['retriever'] = df_mean['top_n_chunks'].map(lambda n: 'sem retriever' if int(n) == 0 else 'com retriever')

    tend_order = ['esquerda', 'neutro', 'direita']
    tend_labels = {'esquerda': 'Left-Wing User', 'neutro': 'No-Context User', 'direita': 'Right-Wing User'}
    df_mean['tendencia_label'] = df_mean['tendencia'].map(tend_labels)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=df_mean,
        x='tendencia_label',
        y='ipi_mean',
        hue='retriever',
        order=[tend_labels[t] for t in tend_order],
        ax=ax,
    )

    # Anotar valores exatos em cada coluna
    for container in ax.containers:
        labels_vals = []
        for bar in container:
            h = bar.get_height()
            if pd.isna(h):
                labels_vals.append("")
            else:
                labels_vals.append(f"{h:.2f}")
        ax.bar_label(container, labels=labels_vals, padding=3, fontsize=11)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
    ax.set_xlabel('User Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Ideological Position Index (IPI)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    ax.legend(title=None, fontsize=12, frameon=False)
    ax.set_title('Mean IPI by User Type: sem vs com retriever', fontsize=15, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'ipi_media_tendencias_pre_pos_retriever', cfg)
    plt.close()


def plot_rag_main_effect_ci(df_ip: pd.DataFrame, cfg: DictConfig):
    if df_ip.empty or 'top_k' not in df_ip.columns or 'rag_relevante' not in df_ip.columns:
        return

    df_use = df_ip.copy()
    df_use['condicao'] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use['condicao'].notna()]
    if df_use.empty:
        return

    df_ci = _compute_ci_from_ip(df_use, group_cols=['modelo', 'condicao'])
    if df_ci.empty:
        return

    df_ci = df_ci.rename(columns={'chameleon_index': 'ci'})
    summary = df_ci.groupby('condicao')['ci'].agg(['mean', 'std', 'count']).reset_index()
    summary['sem'] = summary['std'] / summary['count'].clip(lower=1).pow(0.5)
    summary['ci95'] = 1.96 * summary['sem']

    order = [c for c in _RAG_CONDITION_ORDER if c in summary['condicao'].tolist()]
    summary['condicao'] = pd.Categorical(summary['condicao'], categories=order, ordered=True)
    summary = summary.sort_values('condicao')

    colors = ["#5a6f80" if c != "Baseline" else "#2f3b45" for c in summary['condicao']]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        summary['condicao'],
        summary['mean'],
        yerr=summary['ci95'],
        color=colors,
        capsize=4,
        alpha=0.9,
    )
    ax.set_ylabel('Chameleon Index (CI)', fontsize=14, fontweight='bold')
    ax.set_xlabel('RAG Condition', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.tight_layout()
    save_fig(fig, 'rag_main_effect_ci', cfg)
    plt.close()


def plot_rag_ipi_dumbbell(df_ip: pd.DataFrame, cfg: DictConfig):
    if df_ip.empty or 'top_k' not in df_ip.columns or 'rag_relevante' not in df_ip.columns:
        return

    df_use = df_ip.copy()
    df_use['condicao'] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use['condicao'].isin(["Baseline", "Top-3 Rel", "Top-3 Irrel"])].copy()
    if df_use.empty:
        return

    df_mean = (
        df_use.groupby(['condicao', 'tendencia'])['indice_polarizacao']
        .mean()
        .reset_index()
    )

    cond_order = ["Baseline", "Top-3 Rel", "Top-3 Irrel"]
    tend_order = ["esquerda", "neutro", "direita"]
    colors = {"esquerda": "#e74c3c", "neutro": "#95a5a6", "direita": "#3498db"}
    labels = {"esquerda": "Left-Wing User", "neutro": "No-Context User", "direita": "Right-Wing User"}

    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = {c: i for i, c in enumerate(cond_order)}

    for cond in cond_order:
        subset = df_mean[df_mean['condicao'] == cond]
        if subset.empty:
            continue
        xs = [
            subset[subset['tendencia'] == t]['indice_polarizacao'].mean()
            for t in tend_order
        ]
        ax.plot(xs, [y_positions[cond]] * len(xs), color='#b0b0b0', linewidth=2, zorder=1)
        for t, x in zip(tend_order, xs):
            ax.scatter(x, y_positions[cond], color=colors[t], s=120, zorder=2)

    ax.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(["Baseline (No RAG)", "Top-3 Relevant", "Top-3 Irrelevant"], fontsize=12)
    ax.set_xlabel('Ideological Position Index (IPI)', fontsize=14, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[t], markersize=10, label=labels[t])
        for t in tend_order
    ]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    plt.tight_layout()
    save_fig(fig, 'rag_ipi_dumbbell', cfg)
    plt.close()


def plot_rag_topic_delta_ci(df_pares: pd.DataFrame, cfg: DictConfig):
    if df_pares.empty or 'top_k' not in df_pares.columns or 'rag_relevante' not in df_pares.columns:
        return

    topic_map = {
        'Políticas Sociais': 'Social Policies',
        'Economia': 'Economy',
        'Segurança Pública': 'Public Security',
        'Meio Ambiente': 'Environment',
        'Instituições Democráticas': 'Democratic Institutions',
        'Corrupção e Justiça': 'Corruption and Justice',
        'Educação e Cultura': 'Education and Culture',
    }

    df_use = df_pares.copy()
    df_use['condicao'] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use['condicao'].isin(['Baseline', 'Top-3 Rel', 'Top-3 Irrel'])].copy()
    if df_use.empty:
        return

    df_ci = _compute_ci_from_ip(
        df_use.rename(columns={'diferenca_R': 'indice_polarizacao'}),
        group_cols=['eixo', 'condicao'],
    )
    if df_ci.empty:
        return

    df_ci = df_ci.rename(columns={'chameleon_index': 'ci'})
    df_pivot = df_ci.pivot_table(index='eixo', columns='condicao', values='ci')
    if 'Baseline' not in df_pivot.columns:
        return

    df_plot = pd.DataFrame(index=df_pivot.index)
    df_plot['Top-3 Relevant vs Baseline'] = df_pivot.get('Top-3 Rel') - df_pivot.get('Baseline')
    df_plot['Top-3 Irrelevant vs Baseline'] = df_pivot.get('Top-3 Irrel') - df_pivot.get('Baseline')
    df_plot = df_plot.rename(index=topic_map)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_plot,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Delta CI'},
        ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    plt.tight_layout()
    save_fig(fig, 'rag_topic_delta_ci', cfg)
    plt.close()

def plot_topic_variation(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Topic-Level Variation.
    Heatmap showing Chameleon Index per model and topic (axis).
    """
    topic_translation = {
        'Políticas Sociais': 'Welfare',
        'Segurança Pública': 'Security',
        'Economia': 'Economy',
        'Meio Ambiente': 'Environment',
        'Educação e Cultura': 'Education and Culture',
        'Corrupção e Justiça': 'Corruption and Justice',
        'Instituições Democráticas': 'Democratic Institutions'
    }
    
    df_topic = df_pares.groupby(['modelo', 'eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    
    # Pivot to get neutral, left, right for each model-topic combination
    df_topic_pivot = df_topic.pivot_table(
        index=['modelo', 'eixo'], 
        columns='tendencia', 
        values='diferenca_R'
    ).reset_index()
    
    # Calculate Chameleon Index per topic
    df_topic_pivot['shift_left'] = abs(df_topic_pivot['esquerda'] - df_topic_pivot['neutro'])
    df_topic_pivot['shift_right'] = abs(df_topic_pivot['direita'] - df_topic_pivot['neutro'])
    df_topic_pivot['chameleon_index'] = df_topic_pivot['shift_left'] + df_topic_pivot['shift_right']
    
    # Create heatmap matrix
    heatmap_data = df_topic_pivot.pivot(
        index='modelo', 
        columns='eixo', 
        values='chameleon_index'
    )
    
    # Translate column names to English
    heatmap_data.columns = [topic_translation.get(col, col) for col in heatmap_data.columns]
    
    # Sort rows by average chameleon index across all topics
    heatmap_data['avg'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('avg', ascending=True).drop('avg', axis=1)
    
    # Sort columns (topics) by average chameleon index across all models
    topic_avg = heatmap_data.mean(axis=0).sort_values(ascending=True)
    heatmap_data = heatmap_data[topic_avg.index]
    
    # Create figure (size adjusted for all models)
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Chameleon Index (CI)'}, 
                linewidths=0.5, linecolor='white', vmin=0, vmax=8)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Chameleon Index (CI)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
        
    # Labels and title
    ax.set_xlabel('Topic', fontsize=19, fontweight='bold')
    ax.set_ylabel('Model', fontsize=19, fontweight='bold')
    ax.tick_params(axis='y', labelsize=19)
    ax.tick_params(axis='x', labelsize=19) 
    for text in ax.texts:
        text.set_fontsize(15)
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_fig(fig, "topic_variation", cfg)
    plt.close()

def plot_likert_distribution(df_validos: pd.DataFrame, cfg: DictConfig):
    """
    Response Distribution (Likert Scale).
    Grouped histogram showing response counts across Likert scale for 3 user types.
    """
    # Define Likert scale mapping
    likert_labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    likert_values = [-2, -1, 0, 1, 2]
    
    # Count responses by tendency and score
    df_counts = df_validos.groupby(['tendencia', 'pontuacao']).size().reset_index(name='count')
    
    # Pivot for easier plotting
    df_pivot = df_counts.pivot(index='pontuacao', columns='tendencia', values='count').fillna(0)
    
    # Reindex to ensure all Likert values are present
    df_pivot = df_pivot.reindex(likert_values, fill_value=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define bar width and positions
    bar_width = 0.25
    x_pos = np.arange(len(likert_values))
    
    # Define colors and labels
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    
    # Plot bars for each tendency
    tendencies = ['neutro', 'esquerda', 'direita']
    for i, tend in enumerate(tendencies):
        if tend in df_pivot.columns:
            offset = (i - 1) * bar_width
            ax.bar(x_pos + offset, df_pivot[tend], bar_width, 
                  label=labels[tend], color=colors[tend], alpha=0.85)
    
    # Labels and formatting
    ax.set_xlabel('Likert Scale Response', fontsize=17, fontweight='bold')
    ax.set_ylabel('Response Count', fontsize=17, fontweight='bold')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(likert_labels)
    #Tamanho do eixo x
    ax.tick_params(axis='x', labelsize=19)
    # Add legend
    ax.legend(loc='upper left', fontsize=15)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    save_fig(fig, "likert_distribution", cfg)
    plt.close()

def plot_topic_dot_panel(df_pares: pd.DataFrame, cfg: DictConfig):
    """
    Topic-Level Dot Plot.
    Y-axis: 7 topics. X-axis: IPI.
    Shows mean (across models) IPI for No-Context, Left, and Right users per topic.
    """
    topic_translation = {
        'Políticas Sociais': 'Welfare',
        'Segurança Pública': 'Security',
        'Economia': 'Economy',
        'Meio Ambiente': 'Environment',
        'Educação e Cultura': 'Education and Culture',
        'Corrupção e Justiça': 'Corruption and Justice',
        'Instituições Democráticas': 'Democratic Institutions'
    }
    
    # Mean IPI per topic and tendency (averaged across models)
    df_topic = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].mean().reset_index()
    df_topic_std = df_pares.groupby(['eixo', 'tendencia'])['diferenca_R'].std().reset_index()
    df_topic_std.columns = ['eixo', 'tendencia', 'std']
    df_topic = df_topic.merge(df_topic_std, on=['eixo', 'tendencia'])
    
    # Translate topic names
    df_topic['topic'] = df_topic['eixo'].map(topic_translation)
    
    # Sort topics by the spread (right - left mean) for visual clarity
    topic_order_df = df_topic.pivot(index='topic', columns='tendencia', values='diferenca_R').reset_index()
    topic_order_df['spread'] = abs(topic_order_df['direita'] - topic_order_df['esquerda'])
    topic_order = topic_order_df.sort_values('spread', ascending=True)['topic'].tolist()
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y_pos = np.arange(len(topic_order))
    colors = {'neutro': '#95a5a6', 'esquerda': '#e74c3c', 'direita': '#3498db'}
    labels = {'neutro': 'No-Context User', 'esquerda': 'Left-Wing User', 'direita': 'Right-Wing User'}
    markers = {'neutro': 's', 'esquerda': 'o', 'direita': 'D'}
    
    for tend in ['esquerda', 'neutro', 'direita']:
        subset = df_topic[df_topic['tendencia'] == tend].copy()
        subset['y'] = subset['topic'].map({t: i for i, t in enumerate(topic_order)})
        subset = subset.sort_values('y')
        ax.errorbar(subset['diferenca_R'], subset['y'], xerr=subset['std'],
                    fmt=markers[tend], markersize=14, color=colors[tend],
                    ecolor=colors[tend], alpha=0.8, label=labels[tend],
                    capsize=4, capthick=1.5, zorder=3, linewidth=0)
    
    # Connect left-right with gray lines per topic
    for i, topic in enumerate(topic_order):
        vals = df_topic[df_topic['topic'] == topic]
        left_val = vals[vals['tendencia'] == 'esquerda']['diferenca_R'].values
        right_val = vals[vals['tendencia'] == 'direita']['diferenca_R'].values
        if len(left_val) > 0 and len(right_val) > 0:
            ax.plot([left_val[0], right_val[0]], [i, i], color='gray', alpha=0.3, linewidth=1.5, zorder=1)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topic_order, fontsize=18)
    ax.set_xlabel('Ideological Position Index (IPI)', fontsize=19, fontweight='bold')
    ax.set_ylabel('Topic', fontsize=19, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=17, ncol=3, frameon=False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    save_fig(fig, "topic_dot_panel", cfg)
    plt.close()
