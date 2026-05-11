# RAG New Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new RAG figures (main effect CI bar chart, IPI dumbbell, topic delta CI heatmap) using `top_k`, `rag_relevante`, and `rag_url`, and call them from `run_analysis.py`.

**Architecture:** Add small data-prep helpers in `plotting.py` to build condition labels and aggregates from `df_ip`/`df_pares`. Each figure will use existing `save_fig` and consistent styling. Add unit tests for helpers to keep logic correct without testing matplotlib rendering.

**Tech Stack:** Python 3.12, pandas, seaborn/matplotlib, unittest

---

## File Structure

**Modify**
- `src/analysis/plotting.py`: add helper functions + three plot functions.
- `run_analysis.py`: call the new plotting functions.

**Create**
- `tests/test_plotting_rag_figures.py`: unit tests for data-prep helpers.

---

### Task 1: Condition mapping helpers

**Files:**
- Create: `tests/test_plotting_rag_figures.py`
- Modify: `src/analysis/plotting.py`

- [ ] **Step 1: Write failing tests for condition labeling**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_plotting_rag_figures.py -v`

Expected: FAIL with `ImportError` or `AttributeError` (helper missing).

- [ ] **Step 3: Implement `_map_rag_condition` and ordering helper**

Add to `src/analysis/plotting.py` (near other helpers):

```python
_RAG_CONDITION_ORDER = [
    "Baseline",
    "Top-1 Rel",
    "Top-3 Rel",
    "Top-5 Rel",
    "Top-3 Irrel",
]


def _map_rag_condition(row: dict) -> str | None:
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_plotting_rag_figures.py -v`

Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_plotting_rag_figures.py src/analysis/plotting.py
git commit -m "test: add rag condition mapping helper"
```

---

### Task 2: CI aggregation helpers

**Files:**
- Modify: `tests/test_plotting_rag_figures.py`
- Modify: `src/analysis/plotting.py`

- [ ] **Step 1: Add failing tests for CI aggregation**

Append to `tests/test_plotting_rag_figures.py`:

```python
from src.analysis.plotting import _compute_ci_by_condition


class RagCiAggregationTests(unittest.TestCase):
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests/test_plotting_rag_figures.py -v`

Expected: FAIL (helper missing).

- [ ] **Step 3: Implement `_compute_ci_by_condition`**

Add to `src/analysis/plotting.py`:

```python
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

    df_pivot["shift_left"] = (df_pivot["esquerda"] - df_pivot["neutro"]).abs()
    df_pivot["shift_right"] = (df_pivot["direita"] - df_pivot["neutro"]).abs()
    df_pivot["ci"] = df_pivot["shift_left"] + df_pivot["shift_right"]

    return df_pivot[["condicao", "ci"]]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m unittest tests/test_plotting_rag_figures.py -v`

Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_plotting_rag_figures.py src/analysis/plotting.py
git commit -m "feat: add rag CI aggregation helper"
```

---

### Task 3: Plot Figure 1 (main effect CI)

**Files:**
- Modify: `src/analysis/plotting.py`

- [ ] **Step 1: Add plotting function**

```python
def plot_rag_main_effect_ci(df_ip: pd.DataFrame, cfg: DictConfig):
    if df_ip.empty or "top_k" not in df_ip.columns or "rag_relevante" not in df_ip.columns:
        return

    df_use = df_ip.copy()
    df_use["condicao"] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use["condicao"].notna()]

    ci_by_model = _compute_ci_from_ip(df_use, group_cols=["modelo", "condicao"])
    if ci_by_model.empty:
        return

    ci_by_model = ci_by_model.rename(columns={"chameleon_index": "ci"})
    summary = ci_by_model.groupby("condicao")["ci"].agg(["mean", "std", "count"]).reset_index()
    summary["sem"] = summary["std"] / summary["count"].clip(lower=1).pow(0.5)
    summary["ci95"] = 1.96 * summary["sem"]

    order = [c for c in _RAG_CONDITION_ORDER if c in summary["condicao"].tolist()]
    summary["condicao"] = pd.Categorical(summary["condicao"], categories=order, ordered=True)
    summary = summary.sort_values("condicao")

    colors = ["#5a6f80" if c != "Baseline" else "#2f3b45" for c in summary["condicao"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(
        summary["condicao"],
        summary["mean"],
        yerr=summary["ci95"],
        color=colors,
        capsize=4,
        alpha=0.9,
    )
    ax.set_ylabel("Chameleon Index (CI)", fontsize=14, fontweight="bold")
    ax.set_xlabel("RAG Condition", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    plt.tight_layout()
    save_fig(fig, "figure_rag_main_effect_ci", cfg)
    plt.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/analysis/plotting.py
git commit -m "feat: add main effect CI plot"
```

---

### Task 4: Plot Figure 2 (IPI dumbbell)

**Files:**
- Modify: `src/analysis/plotting.py`

- [ ] **Step 1: Add plotting function**

```python
def plot_rag_ipi_dumbbell(df_ip: pd.DataFrame, cfg: DictConfig):
    if df_ip.empty or "top_k" not in df_ip.columns or "rag_relevante" not in df_ip.columns:
        return

    df_use = df_ip.copy()
    df_use["condicao"] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use["condicao"].isin(["Baseline", "Top-3 Rel", "Top-3 Irrel"])].copy()
    if df_use.empty:
        return

    df_mean = (
        df_use.groupby(["condicao", "tendencia"])["indice_polarizacao"]
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
        subset = df_mean[df_mean["condicao"] == cond]
        if subset.empty:
            continue
        xs = [
            subset[subset["tendencia"] == t]["indice_polarizacao"].mean()
            for t in tend_order
        ]
        ax.plot(xs, [y_positions[cond]] * len(xs), color="#b0b0b0", linewidth=2, zorder=1)
        for t, x in zip(tend_order, xs):
            ax.scatter(x, y_positions[cond], color=colors[t], s=120, zorder=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(["Baseline (No RAG)", "Top-3 Relevant", "Top-3 Irrelevant"], fontsize=12)
    ax.set_xlabel("Ideological Position Index (IPI)", fontsize=14, fontweight="bold")
    ax.set_xlim(-4, 4)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[t], markersize=10, label=labels[t]) for t in tend_order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

    plt.tight_layout()
    save_fig(fig, "figure_rag_ipi_dumbbell", cfg)
    plt.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/analysis/plotting.py
git commit -m "feat: add IPI dumbbell plot"
```

---

### Task 5: Plot Figure 3 (Topic delta CI heatmap)

**Files:**
- Modify: `src/analysis/plotting.py`

- [ ] **Step 1: Add plotting function**

```python
def plot_rag_topic_delta_ci(df_pares: pd.DataFrame, cfg: DictConfig):
    if df_pares.empty or "top_k" not in df_pares.columns or "rag_relevante" not in df_pares.columns:
        return

    topic_map = {
        "Políticas Sociais": "Social Policies",
        "Economia": "Economy",
        "Segurança Pública": "Public Security",
        "Meio Ambiente": "Environment",
        "Instituições Democráticas": "Democratic Institutions",
        "Corrupção e Justiça": "Corruption and Justice",
        "Educação e Cultura": "Education and Culture",
    }

    df_use = df_pares.copy()
    df_use["condicao"] = df_use.apply(_map_rag_condition, axis=1)
    df_use = df_use[df_use["condicao"].isin(["Baseline", "Top-3 Rel", "Top-3 Irrel"])].copy()
    if df_use.empty:
        return

    df_ci = _compute_ci_from_ip(
        df_use.rename(columns={"diferenca_R": "indice_polarizacao"}),
        group_cols=["eixo", "condicao"],
    )
    if df_ci.empty:
        return

    df_ci = df_ci.rename(columns={"chameleon_index": "ci"})
    baseline = df_ci[df_ci["condicao"] == "Baseline"]["ci"]
    if baseline.empty:
        return

    df_ci["delta"] = df_ci["ci"] - df_ci[df_ci["condicao"] == "Baseline"]["ci"].reindex(df_ci.index, fill_value=baseline.mean())

    df_pivot = df_ci.pivot_table(index="eixo", columns="condicao", values="delta")
    df_pivot = df_pivot.rename(index=topic_map)

    df_plot = df_pivot[["Top-3 Rel", "Top-3 Irrel"]].copy()
    df_plot.columns = ["Top-3 Relevant vs Baseline", "Top-3 Irrelevant vs Baseline"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_plot,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Delta CI"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    save_fig(fig, "figure_rag_topic_delta_ci", cfg)
    plt.close()
```

- [ ] **Step 2: Commit**

```bash
git add src/analysis/plotting.py
git commit -m "feat: add topic delta CI heatmap"
```

---

### Task 6: Wire plots into run_analysis

**Files:**
- Modify: `run_analysis.py`

- [ ] **Step 1: Add plot calls**

Add after existing retriever comparison plots:

```python
    plotting.plot_rag_main_effect_ci(df_ip, cfg)
    plotting.plot_rag_ipi_dumbbell(df_ip, cfg)
    plotting.plot_rag_topic_delta_ci(df_pares, cfg)
```

- [ ] **Step 2: Commit**

```bash
git add run_analysis.py
git commit -m "feat: add new RAG figures to analysis run"
```

---

## Plan Self-Review

- Spec coverage: condition mapping, main effect CI, dumbbell IPI, delta CI heatmap, run_analysis calls.
- No placeholders: all steps include explicit code and commands.
- Names consistent: `_map_rag_condition`, `plot_rag_main_effect_ci`, `plot_rag_ipi_dumbbell`, `plot_rag_topic_delta_ci`.
