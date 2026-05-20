# Ideological RAG

Research project for evaluating how context retrieved from Wikipedia changes the ideological behavior of language models on Brazilian political statements.

The experiment compares LLM responses across conditions without retrieval, with relevant retrieval, and with irrelevant retrieval. Responses are collected on a Likert scale and then converted into metrics such as the **Ideological Position Index (IPI)** and the **Chameleon Index (CI)**.

## Objective

This project investigates whether adding external context through RAG reduces, increases, or reorganizes models' ideological adaptation when they answer pairs of political statements.

Each model answers the same statements under three user profiles:

- a user identified with the political left;
- a user identified with the political right;
- a neutral user, with no explicit ideological context.

For each statement, the model must answer with exactly one of the following options:

- `Discordo fortemente`
- `Discordo`
- `Neutro`
- `Concordo`
- `Concordo fortemente`

## Experimental Design

The input data is stored in `dados/pairs.json`. Each item contains:

- `pair_id`: pair identifier;
- `eixo`: thematic axis;
- `p_minus`: statement associated with one position;
- `p_plus`: opposing statement;
- `wiki`: Wikipedia page used to retrieve relevant context.

The RAG conditions are defined in `conf/config.yaml`:

- `top_k = 0`: baseline without retrieval;
- `top_k = 1, 3, 5`: relevant retrieval from the pair's Wikipedia page;
- `top_k = 3` with irrelevant URLs: irrelevant-context controls.

The Wikipedia FAISS index is stored in `wiki_faiss_store/` and used by `src/main/wiki_retrieval.py`.

## Metrics

### Ideological Position Index (IPI)

IPI is computed from the average difference between responses to each `P+` and `P-` pair, aggregated by model, ideological tendency, and RAG condition.

### Chameleon Index (CI)

CI measures how much a model's response changes as a function of the user's ideological profile. It is computed as:

```text
CI = |IPI_left - IPI_neutral| + |IPI_right - IPI_neutral|
```

A higher CI indicates greater model sensitivity to the user's ideological framing.

## Structure

```text
.
├── main.py                         # Collects model responses with/without RAG
├── run_analysis.py                 # Processes responses and generates figures
├── validate_pairs.py               # Auxiliary pair validation
├── merge_caches.py                 # Utility for merging caches
├── conf/
│   ├── config.yaml                 # Models, prompts, RAG settings, collection files
│   ├── analysis_config.yaml        # Analysis and figure settings
│   └── validate_config.yaml        # Validation settings
├── dados/
│   ├── pairs.json                  # Political statement pairs
│   ├── respostas.csv               # Collected responses
│   └── cache*.pkl                  # API call caches
├── src/
│   ├── main/
│   │   ├── utils.py                # API clients, cache, response validation
│   │   └── wiki_retrieval.py       # FAISS retrieval + embeddings
│   └── analysis/
│       ├── processing.py           # Cleaning and IPI computation
│       ├── plotting.py             # Figures
│       └── statistics.py           # Auxiliary statistics
├── tests/                          # Unit tests
├── wiki_faiss_store/               # Local Wikipedia index
└── analises_figures/               # Figures in PNG, SVG, and PDF
```

## Installation

The project uses Pixi to manage the environment.

```bash
pixi install
```

Main dependencies:

- Python 3.12
- pandas
- scipy
- seaborn/matplotlib
- hydra-core
- openai
- google-genai
- xai-sdk
- faiss-cpu

## Environment Variables

Create a `.env` file with the keys required by the providers used in `conf/config.yaml`:

```text
OPEN_AI_API_KEY=...
DEEPINFRA_API_KEY=...
GEMINI_API_KEY=...
GROK_API_KEY=...
MARITACA_API_KEY=...
OLLAMA_HOST=...
```

`DEEPINFRA_API_KEY` is also used for query embeddings in Wikipedia retrieval when `WIKI_EMBEDDING_MODEL` points to a model served by DeepInfra.

## How to Run

### Collect responses

```bash
pixi run start
```

This command runs `main.py`, reads `dados/pairs.json`, queries the models defined in `conf/config.yaml`, and saves:

- responses to `dados/respostas.csv`;
- cache data to `dados/cache.pkl`.

### Generate analyses and figures

```bash
pixi run analises
```

This command runs `run_analysis.py`, reads `dados/respostas.csv`, and saves figures to:

- `analises_figures/png/`
- `analises_figures/svg/`
- `analises_figures/pdf/`

## Main Configuration

In `conf/config.yaml`:

- `MODELOS_A_AVALIAR`: `[model, provider]` pairs;
- `TEMPERATURES`: evaluated temperatures;
- `REPETICOES_POR_TEMP`: repetitions per temperature;
- `ARQUIVO_PERGUNTAS`: JSON file with political statement pairs;
- `ARQUIVO_CACHE`: response cache;
- `ARQUIVO_SAIDA`: final CSV output;
- `PROMPT_ESQUERDA`, `PROMPT_DIREITA`, `PROMPT_NEUTRO`: user profiles;
- `TOP_N_CHUNKS`: relevant-retrieval conditions;
- `WIKI_STORE_DIR`: FAISS index directory;
- `WIKI_EMBEDDING_MODEL`: embedding model for queries;
- `WIKI_IRRELEVANT_URLS`: pages used as irrelevant controls.

In `conf/analysis_config.yaml`:

- `paths.input_file`: CSV used for analysis;
- `paths.output_dir`: figure output directory;
- `analysis.likert_map`: Likert-to-numeric mapping;
- `analysis.save_plots`: controls whether figures are saved.

## Supported Providers

The code currently supports calls to:

- DeepInfra
- OpenAI
- Google Gemini
- xAI/Grok
- Maritaca
- Ollama

The provider for each model is defined in the second position of each entry in `MODELOS_A_AVALIAR`.

## Notes

- Collection is asynchronous and uses provider-specific semaphores to control concurrency.
- The cache includes the RAG context in the hash, preventing response reuse when the retrieved context changes.
- Relevant retrieval is restricted to the Wikipedia page associated with the statement pair; irrelevant controls use manually defined URLs.
- Figures and statistics are derived from valid responses after Likert mapping.

## Authors

Anderson Soares (a241149@dac.unicamp.br)
