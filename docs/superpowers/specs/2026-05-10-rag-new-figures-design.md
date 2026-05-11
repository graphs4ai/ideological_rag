# RAG figures for main effect, IPI dumbbell, and topic delta CI (design)

Date: 2026-05-10

## Summary
Add three new figures based on new RAG columns (`top_k`, `rag_relevante`, `rag_url`) to quantify main effect of RAG on CI, IPI spectrum contraction, and topic-level delta CI. Implement in existing plotting module and call from run_analysis.

## Goals
- Use new RAG columns for condition selection.
- Produce three new figures matching existing Figure 1 styling.
- Ensure outputs are saved through existing `save_fig` flow.

## Non-goals
- Refactoring existing plotting logic.
- Changing existing analysis output formats.

## Conditions
Conditions are derived from `top_k` and `rag_relevante`:
- Baseline: `top_k=0`
- Top-1 Rel: `top_k=1` and `rag_relevante=true`
- Top-3 Rel: `top_k=3` and `rag_relevante=true`
- Top-5 Rel: `top_k=5` and `rag_relevante=true`
- Top-3 Irrel: `top_k=3` and `rag_relevante=false`

## Figure 1: Average CI by RAG condition (main effect)
- Y-axis: Chameleon Index (CI), mean across all models/pairs.
- X-axis: Baseline, Top-1 Rel, Top-3 Rel, Top-5 Rel, Top-3 Irrel.
- Bars with 95% confidence intervals; neutral color, baseline darker.

## Figure 2: Dumbbell plot (IPI spectrum contraction)
- Y-axis: Baseline (No RAG), Top-3 Relevant, Top-3 Irrelevant.
- X-axis: IPI from -4 to +4 with vertical dashed line at 0.
- Each condition shows three points (left/neutral/right) connected by a gray line.
- Colors: red (left), gray (neutral), blue (right). Legend included.

## Figure 3: Heatmap (Topic-level delta CI)
- Compute CI per topic and condition, then delta vs baseline.
- Y-axis topics (English): Social Policies, Economy, Public Security, Environment,
  Democratic Institutions, Corruption and Justice, Education and Culture.
- X-axis: Top-3 Relevant vs Baseline; Top-3 Irrelevant vs Baseline.
- Diverging colormap (coolwarm/RdBu), annotate values with 2 decimals.

## Implementation
- Add three new plot functions to `src/analysis/plotting.py`.
- Call these functions from `run_analysis.py` after current retriever comparison plots.
- Save names:
  - `figure_rag_main_effect_ci`
  - `figure_rag_ipi_dumbbell`
  - `figure_rag_topic_delta_ci`

## Data requirements
- Use `top_k`, `rag_relevante`, `rag_url` from `dados/respostas.csv`.
- Use `diferenca_R` and derived IPI/CI from `processing` outputs.

## Open questions
None.
