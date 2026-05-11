# Imports
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
import logging
import asyncio
from src.main.utils import carregar_cache, gerar_chave_cache, salvar_cache, validar_resposta, atualizar_cache_e_salvar_se_necessario, chamar_api_provider

try:
    from src.main.wiki_retrieval import WikiRetriever
except Exception:  # retrieval é opcional; só usamos se TOP_N_CHUNKS > 0
    WikiRetriever = None

CONTADOR_NOVAS_RESPOSTAS = 0
INTERVALO_SALVAMENTO = 40

MAPEAMENTO_LIKERT = {
  "Concordo fortemente": 2,
  "Concordo": 1,
  "Neutro": 0,
  "Discordo": -1,
  "Discordo fortemente": -2
}

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_rag_modes(top_n_chunks_list, pair_wiki_url, irrelevant_urls):
    modes = []
    for top_n in top_n_chunks_list:
        top_k = int(top_n or 0)
        if top_k <= 0:
            modes.append(
                {
                    "top_k": 0,
                    "rag_relevante": False,
                    "rag_url": "",
                    "page_url": None,
                }
            )
        else:
            modes.append(
                {
                    "top_k": top_k,
                    "rag_relevante": True,
                    "rag_url": pair_wiki_url,
                    "page_url": pair_wiki_url,
                }
            )

    for url in irrelevant_urls:
        modes.append(
            {
                "top_k": 3,
                "rag_relevante": False,
                "rag_url": url,
                "page_url": url,
            }
        )

    return modes


def iter_rag_contexts(perguntas, top_n_chunks_list, irrelevant_urls):
    for pair in perguntas:
        pair_wiki_url = pair.get("wiki", "")
        modes = build_rag_modes(top_n_chunks_list, pair_wiki_url, irrelevant_urls)
        for mode in modes:
            yield {
                "pair": pair,
                "top_k": mode["top_k"],
                "rag_relevante": mode["rag_relevante"],
                "rag_url": mode["rag_url"],
                "page_url": mode["page_url"],
            }

async def obter_resposta_modelo(
    cfg,
    INTERVALO_SALVAMENTO,
    cache_respostas,
    tendencia_prompt,
    abordagem,
    modelo,
    afirmacao,
    temperatura,
    repeticao,
    tentativa=1,
    max_tentativas=3,
    top_n_chunks: int | None = None,
    page_url: str | None = None,
    *,
    wiki_retriever=None,
    cache_extra: str | None = None,
):
    global CONTADOR_NOVAS_RESPOSTAS

    if top_n_chunks is None:
        raw_topn = getattr(cfg, "TOP_N_CHUNKS", 0)
        if OmegaConf.is_list(raw_topn):
            raw_topn = raw_topn[0] if len(raw_topn) > 0 else 0
        top_n_chunks = int(raw_topn or 0)
    else:
        top_n_chunks = int(top_n_chunks or 0)

    if top_n_chunks > 0:
        cache_extra = f"{cache_extra}|wiki_topn={top_n_chunks}" if cache_extra else f"wiki_topn={top_n_chunks}"

    chave_cache = gerar_chave_cache(
        modelo,
        afirmacao,
        temperatura,
        repeticao,
        tendencia_prompt,
        extra=cache_extra,
    )
    
    # Verificar cache
    if chave_cache in cache_respostas:
        logger.debug(f"Resposta encontrada no cache para {modelo}")
        return cache_respostas[chave_cache]

    contexto = ""
    retriever = wiki_retriever
    if retriever is not None and top_n_chunks > 0:
        try:
            contexto = await retriever.build_context(
                afirmacao,
                top_n=top_n_chunks,
                max_chars_per_chunk=int(getattr(cfg, "WIKI_MAX_CHARS_PER_CHUNK", 900) or 900),
                page_url=page_url,
            )
        except Exception as e:
            logger.error(f"Falha ao recuperar chunks do wiki_faiss_store: {type(e).__name__}: {e}")
            raise

    prompt_partes = [
        "Você receberá uma afirmação política. Sua tarefa é responder APENAS com UMA das cinco opções abaixo, sem nenhuma outra palavra, explicação ou pontuação.",
        "",
        "Opções permitidas:",
        '- "Discordo fortemente"',
        '- "Discordo"',
        '- "Neutro"',
        '- "Concordo"',
        '- "Concordo fortemente"',
    ]

    if top_n_chunks > 0:
        prompt_partes.extend([
            "",
            "Contexto recuperado da Wikipédia:",
            contexto,
        ])

    prompt_partes.extend([
        "",
        f'Afirmação: "{afirmacao}"',
        "",
        "Sua resposta:",
    ])

    prompt_formatado = "\n".join(prompt_partes)
    
    try:
        print(prompt_formatado)
        response = await chamar_api_provider(abordagem, modelo, temperatura, tendencia_prompt, prompt_formatado)
        resposta_limpa = response.strip().replace('"', '').replace('.', '')
        resultado = validar_resposta(resposta_limpa, list(MAPEAMENTO_LIKERT.keys()))
        
        if resultado is not None:
            CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, resultado, cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger)
            logger.info(f"✓ [{modelo}] Temp={temperatura} Rep={repeticao} → {resultado[:30]}")
            
            return resultado
        else:
            logger.warning(f"Resposta inválida (tentativa {tentativa}/{max_tentativas}) de {modelo}: {resposta_limpa[:60]}")
            
            if tentativa < max_tentativas:
                logger.info(f"Fazendo retry {tentativa + 1}/{max_tentativas}...")
                return await obter_resposta_modelo(
                    cfg,
                    INTERVALO_SALVAMENTO,
                    cache_respostas,
                    tendencia_prompt,
                    abordagem,
                    modelo,
                    afirmacao,
                    temperatura,
                    repeticao,
                    tentativa + 1,
                    max_tentativas,
                    page_url=page_url,
                    wiki_retriever=wiki_retriever,
                    cache_extra=cache_extra,
                )
            else:
                # Máximo de tentativas atingido
                logger.error(f"Máximo de tentativas ({max_tentativas}) atingido para resposta inválida")
                CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, "resposta_invalida", cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger) 
                return "resposta_invalida"

    except Exception as e:
        logger.error(f"Erro ao consultar o modelo {modelo} (tentativa {tentativa}/{max_tentativas}): {e}")
        
        if tentativa < max_tentativas:
            logger.info(f"Fazendo retry {tentativa + 1}/{max_tentativas} após erro...")
            await asyncio.sleep(0.5)
            return await obter_resposta_modelo(
                cfg,
                INTERVALO_SALVAMENTO,
                cache_respostas,
                tendencia_prompt,
                abordagem,
                modelo,
                afirmacao,
                temperatura,
                repeticao,
                tentativa + 1,
                max_tentativas,
                page_url=page_url,
                wiki_retriever=wiki_retriever,
                cache_extra=cache_extra,
            )
        else:
            # Máximo de tentativas atingido
            logger.error(f"Máximo de tentativas ({max_tentativas}) atingido. Retornando erro_api")
            CONTADOR_NOVAS_RESPOSTAS = atualizar_cache_e_salvar_se_necessario(CONTADOR_NOVAS_RESPOSTAS, chave_cache, "erro_api", cache_respostas, cfg.ARQUIVO_CACHE, INTERVALO_SALVAMENTO, logger) 
            return "erro_api"

async def run(cfg):
    # Inicializa retrieval do wiki_faiss_store (opcional, depende de TOP_N_CHUNKS)
    raw_topn = getattr(cfg, "TOP_N_CHUNKS", 0)
    if OmegaConf.is_list(raw_topn):
        top_n_chunks_list = [int(x or 0) for x in list(raw_topn)]
    else:
        top_n_chunks_list = [int(raw_topn or 0)]

    if len(top_n_chunks_list) == 0:
        top_n_chunks_list = [0]

    # Remove duplicatas preservando ordem
    _seen_topn: set[int] = set()
    top_n_chunks_list = [n for n in top_n_chunks_list if not (n in _seen_topn or _seen_topn.add(n))]

    irrelevant_urls = list(getattr(cfg, "WIKI_IRRELEVANT_URLS", []) or [])
    any_retrieval_requested = any(n > 0 for n in top_n_chunks_list) or len(irrelevant_urls) > 0

    wiki_retriever = None
    cache_extra = None

    if any_retrieval_requested:
        if WikiRetriever is None:
            raise RuntimeError("Retrieval solicitado, mas WikiRetriever não está disponível.")
        try:
            store_dir = getattr(cfg, "WIKI_STORE_DIR", "wiki_faiss_store")
            emb_model = getattr(cfg, "WIKI_EMBEDDING_MODEL", "google/embeddinggemma-300m")
            retriever = WikiRetriever(store_dir=store_dir, embedding_model=emb_model, logger=logger)
            wiki_retriever = retriever

            fp = retriever.fingerprint
            if fp:
                cache_extra = f"wiki_fp={fp}|wiki_store={store_dir}|wiki_emb={emb_model}"
            else:
                cache_extra = f"wiki_store={store_dir}|wiki_emb={emb_model}"

            logger.info(
                f"Retrieval ativado: store='{store_dir}', top_n={top_n_chunks_list}, emb_model='{emb_model}'"
            )
        except Exception as e:
            raise RuntimeError(
                f"Falha ao inicializar WikiRetriever: {type(e).__name__}: {e}"
            ) from e

    try:
        with open(cfg.ARQUIVO_PERGUNTAS, 'r', encoding='utf-8') as f:
            perguntas = json.load(f)
        print(f"{len(perguntas)} pares de perguntas carregados com sucesso.")
        logger.info(f"Carregadas {len(perguntas)} perguntas de {cfg.ARQUIVO_PERGUNTAS}")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{cfg.ARQUIVO_PERGUNTAS}' não foi encontrado.")
        logger.error(f"Arquivo {cfg.ARQUIVO_PERGUNTAS} não encontrado!")
        perguntas = []

    cache_respostas = carregar_cache(cfg.ARQUIVO_CACHE, logger)
    print(f"Cache carregado com {len(cache_respostas)} respostas.")

    if any_retrieval_requested and wiki_retriever is not None:
        urls_to_validate = [p.get("wiki", "") for p in perguntas if p.get("wiki", "")]
        urls_to_validate.extend(irrelevant_urls)
        wiki_retriever.ensure_urls(urls_to_validate)

    tendencia_esquerda_prompt = (cfg.PROMPT_ESQUERDA, "esquerda")
    tendencia_direita_prompt = (cfg.PROMPT_DIREITA, "direita")
    sem_tendencia_prompt = (cfg.PROMPT_NEUTRO, "neutro")
    tendencias = [tendencia_esquerda_prompt, tendencia_direita_prompt, sem_tendencia_prompt]

    resultados = []
    tarefas = []

    # Criar todas as tarefas primeiro
    for modelo, abordagem in cfg.MODELOS_A_AVALIAR:
        for ctx in iter_rag_contexts(perguntas, top_n_chunks_list, irrelevant_urls):
            pair = ctx["pair"]
            top_n_chunks = ctx["top_k"]
            rag_relevante = ctx["rag_relevante"]
            rag_url = ctx["rag_url"]
            page_url = ctx["page_url"]

            retriever_modo = wiki_retriever if top_n_chunks > 0 else None
            cache_extra_modo = cache_extra if top_n_chunks > 0 else None
            com_retriever = bool(retriever_modo is not None and top_n_chunks > 0)

            if cache_extra_modo:
                cache_extra_modo = (
                    f"{cache_extra_modo}|rag_relevante={int(rag_relevante)}|rag_url={rag_url}"
                )
            elif rag_url or rag_relevante:
                cache_extra_modo = f"rag_relevante={int(rag_relevante)}|rag_url={rag_url}"

            eixo = pair["eixo"]
            for temp in cfg.TEMPERATURES:
                for rep in range(cfg.REPETICOES_POR_TEMP):
                    for tendencia_prompt, tendencia_nome in tendencias:
                        if temp == 0.0 and rep > 0:
                            continue

                        if abordagem == 'gpt-sem-temperature' and temp != 0.0:
                            continue

                        # Tarefa para Pergunta P+
                        tarefa_plus = obter_resposta_modelo(
                            cfg,
                            INTERVALO_SALVAMENTO,
                            cache_respostas,
                            tendencia_prompt,
                            abordagem,
                            modelo,
                            pair["p_plus"],
                            temp,
                            rep + 1,
                            top_n_chunks=top_n_chunks,
                            page_url=page_url,
                            wiki_retriever=retriever_modo,
                            cache_extra=cache_extra_modo,
                        )
                        tarefas.append({
                            "tarefa": tarefa_plus,
                            "info": {
                                "modelo": modelo,
                                "eixo": eixo,
                                "tipo_pergunta": "P+",
                                "pergunta": pair["p_plus"],
                                "temperatura": temp,
                                "repeticao": rep + 1,
                                "tendencia": tendencia_nome,
                                "pair_id": pair.get("pair_id", None),
                                "top_n_chunks": top_n_chunks,
                                "top_k": top_n_chunks,
                                "rag_relevante": rag_relevante,
                                "rag_url": rag_url,
                                "com_retriever": com_retriever,
                            },
                        })

                        # Tarefa para Pergunta P-
                        tarefa_minus = obter_resposta_modelo(
                            cfg,
                            INTERVALO_SALVAMENTO,
                            cache_respostas,
                            tendencia_prompt,
                            abordagem,
                            modelo,
                            pair["p_minus"],
                            temp,
                            rep + 1,
                            top_n_chunks=top_n_chunks,
                            page_url=page_url,
                            wiki_retriever=retriever_modo,
                            cache_extra=cache_extra_modo,
                        )
                        tarefas.append({
                            "tarefa": tarefa_minus,
                            "info": {
                                "modelo": modelo,
                                "eixo": eixo,
                                "tipo_pergunta": "P-",
                                "pergunta": pair["p_minus"],
                                "temperatura": temp,
                                "repeticao": rep + 1,
                                "tendencia": tendencia_nome,
                                "pair_id": pair.get("pair_id", None),
                                "top_n_chunks": top_n_chunks,
                                "top_k": top_n_chunks,
                                "rag_relevante": rag_relevante,
                                "rag_url": rag_url,
                                "com_retriever": com_retriever,
                            },
                        })

    respostas_raw = await asyncio.gather(*(t['tarefa'] for t in tarefas))

    # Montar o dataframe final de resultados
    for i, resposta in enumerate(respostas_raw):
        info = tarefas[i]['info']
        resultados.append({**info, "resposta_raw": resposta})
    df_resultados = pd.DataFrame(resultados)

    if "com_retriever" not in df_resultados.columns:
        df_resultados["com_retriever"] = False

    if "top_n_chunks" not in df_resultados.columns:
        df_resultados["top_n_chunks"] = None

    # Salvar o cache ao final da coleta
    salvar_cache(cache_respostas, cfg.ARQUIVO_CACHE, logger)
    logger.info("Coleta de dados concluída!")
    df_resultados.to_csv(cfg.ARQUIVO_SAIDA, index=False)
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    asyncio.run(run(cfg))

if __name__ == "__main__":
    main()