"""Fase 3 - Pipeline RAG (Voice of Customer sobre avaliacoes da Olist).

Fluxo:
  pergunta -> processamento (LLM reescreve e checa escopo) -> embeddings
  -> recuperacao no ChromaDB -> limiar de relevancia (>= N evidencias unicas)
  -> selecao diversificada -> contexto numerado -> geracao (Ollama)
  -> guarda de palavras de frequencia (v3) -> checagem de citacoes

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo e o Ollama aberto):
    python src/rag.py "Quais os principais problemas de entrega?"
    python src/rag.py "O que os clientes elogiam?" --faixa positiva --debug
    python src/rag.py                      # modo interativo
"""
import argparse
import os
import re
import time
import unicodedata

import chromadb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLECAO, MODELO_EMBEDDINGS, PREFIXO_PERGUNTA

# --- Parametros do pipeline (justificar no README) -------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = "qwen3:8b"
LIMIAR = 0.85          # similaridade minima para uma avaliacao contar como evidencia
MIN_EVIDENCIAS = 3     # textos unicos acima do limiar para tentar responder
TOP_K_BUSCA = 30       # candidatos por consulta (busca larga, depois filtra e seleciona)
MAX_EVIDENCIAS = 10    # evidencias enviadas ao modelo
NEAR_DUP = 0.93        # similaridade entre evidencias a partir da qual sao "quase iguais"
TIMEOUT_S = 300

MSG_SEM_EVIDENCIA = (
    "Não encontrei evidências suficientes nas avaliações para responder a essa pergunta."
)
MSG_FORA_ESCOPO = (
    "Essa pergunta está fora do escopo desta base, que contém apenas avaliações escritas "
    "por clientes (entrega, produto, atendimento e experiência de compra)."
)

PROMPT_REESCRITA = """Você ajuda a buscar avaliações de clientes de uma loja online brasileira.
Tarefa: dada a PERGUNTA de um gestor, escreva de 3 a 5 frases curtas, no estilo de avaliações \
reais de clientes, que seriam boas respostas para ela. Uma frase por linha, sem numeração e sem explicações.
Se a pergunta NÃO tiver relação com compras, entrega, produtos, vendedores, atendimento ou \
experiência dos clientes, ou pedir números totais, vendas, preços, telefones ou dados \
cadastrais que não aparecem em avaliações, responda apenas: FORA_DO_ESCOPO

PERGUNTA: {pergunta}"""

SISTEMA_RESPOSTA = """Você é um analista de Voice of Customer. Responda à PERGUNTA usando SOMENTE \
as AVALIAÇÕES numeradas fornecidas.
Regras:
1. Use apenas o que está escrito nas avaliações. Não use conhecimento externo, não invente fatos \
e não infira causas ou detalhes que as avaliações não mencionam.
2. Cite as avaliações que sustentam cada afirmação com o número entre colchetes, por exemplo [2][5]. \
Cite no máximo 3 por afirmação, escolhendo as mais representativas.
3. As avaliações são uma amostra recuperada, não uma estatística. Não informe percentuais ou totais \
e não compare a frequência entre temas: nada de "a maioria", "muitos", "o mais citado" ou \
"principal problema". Escreva "há relatos de ...".
4. Se as avaliações não responderem à pergunta, responda exatamente: \
Não encontrei evidências suficientes nas avaliações para responder a essa pergunta.
5. Responda em português do Brasil, em no máximo 8 linhas: uma frase de síntese e depois \
tópicos curtos, um por tema."""


# --- Funcoes auxiliares ------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minusculas, sem acentos nem pontuacao: usado para achar textos repetidos."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def selecionar_diversas(listas: list[list[dict]], limiar: float, maximo: int,
                        near_dup: float = NEAR_DUP) -> list[dict]:
    """Escolhe evidencias variadas.

    Reveza entre as consultas (a original e as frases geradas na reescrita), pegando a
    melhor avaliacao ainda nao usada de cada uma, para que todos os temas aparecam.
    Ignora textos repetidos e avaliacoes quase iguais (cosseno >= near_dup) as ja escolhidas.
    Cada item de `listas` esta ordenado da maior para a menor similaridade.
    """
    filas = [[c for c in lista if c["sim"] >= limiar] for lista in listas]
    posicoes = [0] * len(filas)
    escolhidas, ids, textos = [], set(), set()
    while len(escolhidas) < maximo:
        avancou = False
        for q, fila in enumerate(filas):
            if len(escolhidas) >= maximo:
                break
            while posicoes[q] < len(fila):
                cand = fila[posicoes[q]]
                posicoes[q] += 1
                chave = normalizar(cand["doc"])
                if cand["id"] in ids or chave in textos:
                    continue
                if any(float(cand["emb"] @ e["emb"]) >= near_dup for e in escolhidas):
                    continue
                escolhidas.append(cand)
                ids.add(cand["id"])
                textos.add(chave)
                avancou = True
                break
        if not avancou:
            break
    return escolhidas


def chamar_ollama(mensagens: list[dict], modelo: str, temperatura: float = 0.1) -> str:
    corpo = {
        "model": modelo,
        "messages": mensagens,
        "stream": False,
        "keep_alive": "30m",  # mantem o modelo na memoria entre perguntas (evita ~20 s de recarga)
        "think": False,  # modelos com raciocinio (ex.: Qwen3): resposta direta e mais rapida
        "options": {"temperature": temperatura, "num_ctx": 4096},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=corpo, timeout=TIMEOUT_S)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as erro:
        raise RuntimeError(
            f"Nao consegui conectar ao Ollama em {OLLAMA_URL}. Abra o Ollama e tente de novo."
        ) from erro
    texto = r.json()["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()


def montar_contexto(evidencias: list[dict]) -> str:
    return "\n".join(
        f"[{i}] (nota {e['meta']['nota']}, {e['meta']['ano_mes']}) {e['doc']}"
        for i, e in enumerate(evidencias, start=1)
    )


def conferir_citacoes(resposta: str, total: int) -> tuple[list[int], list[int]]:
    citadas = {int(n) for n in re.findall(r"\[(\d+)\]", resposta)}
    validas = sorted(n for n in citadas if 1 <= n <= total)
    invalidas = sorted(n for n in citadas if not 1 <= n <= total)
    return validas, invalidas


# --- Guarda de palavras de frequencia (v3) ----------------------------------
# A resposta descreve uma amostra recuperada, nao uma estatistica. Estas expressoes
# comparam frequencia sem base e sao detectadas na resposta (texto normalizado).
PADROES_FREQUENCIA = re.compile(
    r"\b(principalmente|principais|principal|maioria|majoritariamente|predominantemente"
    r"|frequentemente|geralmente|comumente|muitos|muitas"
    r"|mais (?:citad[oa]s?|comum|comuns|frequentes?|recorrentes?|mencionad[oa]s?))\b"
)


def detectar_frequencia(resposta: str, pergunta: str = "") -> list[str]:
    """Expressoes de frequencia na resposta. Ignora as que o proprio usuario usou na pergunta
    (ex.: 'principais problemas'), pois ecoar a pergunta nao e afirmar frequencia."""
    pergunta_norm = normalizar(pergunta)
    achadas = []
    for m in PADROES_FREQUENCIA.finditer(normalizar(resposta)):
        termo = m.group(0)
        if termo not in pergunta_norm and termo not in achadas:
            achadas.append(termo)
    return achadas


def pedir_reescrita_sem_frequencia(mensagens: list[dict], resposta: str,
                                   termos: list[str]) -> list[dict]:
    """Mensagens para a segunda tentativa: mostra a resposta anterior e pede a correcao."""
    return mensagens + [
        {"role": "assistant", "content": resposta},
        {"role": "user", "content": (
            "Reescreva a resposta sem estas expressões de frequência: "
            + ", ".join(f'"{t}"' for t in termos)
            + '. Use "há relatos de ..." e não compare a frequência entre temas. '
            "Mantenha o mesmo formato: uma frase de síntese e depois tópicos curtos, um por tema, "
            "cada tópico com as suas próprias citações [n] no fim da frase. "
            "Mantenha todas as outras regras. Responda só com a nova versão."
        )},
    ]


# --- Pipeline ------------------------------------------------------------------
class RAG:
    def __init__(self, modelo_llm=MODELO_LLM, limiar=LIMIAR, min_evidencias=MIN_EVIDENCIAS):
        self.modelo_llm = modelo_llm
        self.limiar = limiar
        self.min_evidencias = min_evidencias
        self.encoder = SentenceTransformer(MODELO_EMBEDDINGS)
        self.colecao = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLECAO)

    # Etapa 1: processamento da pergunta
    def processar_pergunta(self, pergunta: str, reescrever: bool = True):
        """Retorna (consultas, fora_do_escopo). Consultas = pergunta original + frases geradas."""
        consultas = [pergunta]
        if not reescrever:
            return consultas, False
        saida = chamar_ollama(
            [{"role": "user", "content": PROMPT_REESCRITA.format(pergunta=pergunta)}],
            self.modelo_llm,
        )
        if "fora do escopo" in normalizar(saida).replace("_", " "):
            return consultas, True
        for linha in saida.splitlines():
            frase = re.sub(r"^[\-\*\d\.\)\s]+", "", linha).strip()
            if frase:
                consultas.append(frase)
        return consultas[:6], False

    # Etapa 2: recuperacao semantica (uma lista ordenada por consulta)
    def buscar(self, consultas: list[str], faixa: str | None = None) -> list[list[dict]]:
        onde = {"faixa_nota": faixa} if faixa else None
        listas = []
        for texto in consultas:
            vetor = self.encoder.encode(PREFIXO_PERGUNTA + texto, normalize_embeddings=True)
            res = self.colecao.query(
                query_embeddings=[vetor.tolist()],
                n_results=TOP_K_BUSCA,
                where=onde,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            listas.append([
                {"id": id_, "doc": doc, "meta": meta, "sim": 1 - dist,
                 "emb": np.asarray(emb, dtype="float32")}
                for id_, doc, meta, dist, emb in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0],
                    res["distances"][0], res["embeddings"][0])
            ])
        return listas

    def perguntar(self, pergunta: str, faixa: str | None = None, reescrever: bool = True) -> dict:
        resultado = {
            "pergunta": pergunta, "status": None, "resposta": None, "evidencias": [],
            "citacoes_validas": [], "citacoes_invalidas": [], "avisos": [], "detalhes": {},
        }
        tempos = resultado["detalhes"]["tempos_s"] = {}
        inicio = time.perf_counter()

        consultas, fora = self.processar_pergunta(pergunta, reescrever)
        tempos["reescrita"] = round(time.perf_counter() - inicio, 1)
        resultado["detalhes"]["consultas"] = consultas
        if fora:
            tempos["total"] = tempos["reescrita"]
            resultado.update(status="fora_do_escopo", resposta=MSG_FORA_ESCOPO)
            return resultado

        t = time.perf_counter()
        listas = self.buscar(consultas, faixa)
        tempos["busca"] = round(time.perf_counter() - t, 1)

        melhor_por_id: dict[str, dict] = {}
        for lista in listas:
            for c in lista:
                if c["id"] not in melhor_por_id or c["sim"] > melhor_por_id[c["id"]]["sim"]:
                    melhor_por_id[c["id"]] = c
        acima = {i: c for i, c in melhor_por_id.items() if c["sim"] >= self.limiar}
        textos_unicos = {normalizar(c["doc"]) for c in acima.values()}
        melhor = max((lista[0]["sim"] for lista in listas if lista), default=None)
        resultado["detalhes"].update(
            candidatos=len(melhor_por_id),
            acima_do_limiar=len(acima),
            textos_unicos_acima_do_limiar=len(textos_unicos),
            melhor_similaridade=round(melhor, 3) if melhor is not None else None,
        )

        if len(textos_unicos) < self.min_evidencias:
            tempos["total"] = round(time.perf_counter() - inicio, 1)
            resultado.update(status="sem_evidencia", resposta=MSG_SEM_EVIDENCIA)
            return resultado

        evidencias = selecionar_diversas(listas, self.limiar, MAX_EVIDENCIAS)
        resultado["detalhes"]["evidencias_selecionadas"] = len(evidencias)

        mensagens = [
            {"role": "system", "content": SISTEMA_RESPOSTA},
            {"role": "user",
             "content": f"PERGUNTA: {pergunta}\n\nAVALIAÇÕES:\n{montar_contexto(evidencias)}"},
        ]
        t = time.perf_counter()
        resposta = chamar_ollama(mensagens, self.modelo_llm)
        tempos["geracao"] = round(time.perf_counter() - t, 1)

        # Guarda de frequencia: uma nova versao, no maximo; se persistir, aviso
        termos = detectar_frequencia(resposta, pergunta)
        resultado["detalhes"]["frequencia_detectada"] = termos
        if termos:
            t = time.perf_counter()
            resposta = chamar_ollama(
                pedir_reescrita_sem_frequencia(mensagens, resposta, termos), self.modelo_llm)
            tempos["reescrita_frequencia"] = round(time.perf_counter() - t, 1)
            restantes = detectar_frequencia(resposta, pergunta)
            resultado["detalhes"]["frequencia_apos_reescrita"] = restantes
            if restantes:
                resultado["avisos"].append(
                    "A resposta ainda contém expressões de frequência "
                    f"({', '.join(restantes)}). As avaliações são uma amostra recuperada, "
                    "não uma estatística.")
        tempos["total"] = round(time.perf_counter() - inicio, 1)

        if normalizar(resposta).startswith(normalizar(MSG_SEM_EVIDENCIA)[:40]):
            resultado.update(status="sem_evidencia", resposta=MSG_SEM_EVIDENCIA)
            return resultado

        validas, invalidas = conferir_citacoes(resposta, len(evidencias))
        if not validas:
            resultado["avisos"].append("A resposta não cita nenhuma evidência; trate com cautela.")
        if invalidas:
            resultado["avisos"].append(f"A resposta cita números inexistentes: {invalidas}.")
        resultado.update(
            status="respondida", resposta=resposta, evidencias=evidencias,
            citacoes_validas=validas, citacoes_invalidas=invalidas,
        )
        return resultado


# --- Linha de comando ------------------------------------------------------------
def imprimir(res: dict, debug: bool) -> None:
    print("\n" + "=" * 72)
    print(f"Pergunta: {res['pergunta']}")
    print(f"Status:   {res['status']}")
    print("-" * 72)
    print(res["resposta"])
    for aviso in res["avisos"]:
        print(f"\n[aviso] {aviso}")
    if res["evidencias"]:
        print("\nEvidências (* = citada na resposta):")
        for i, e in enumerate(res["evidencias"], start=1):
            marca = "*" if i in res["citacoes_validas"] else " "
            print(f" {marca}[{i}] sim={e['sim']:.3f} | nota {e['meta']['nota']} "
                  f"| {e['meta']['ano_mes']} | {e['doc']}")
    if debug:
        print("\n[debug]")
        for k, v in res["detalhes"].items():
            print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG sobre avaliacoes de clientes (Olist)")
    ap.add_argument("pergunta", nargs="*", help="pergunta (vazio = modo interativo)")
    ap.add_argument("--faixa", choices=["negativa", "neutra", "positiva"],
                    help="filtra por faixa de nota (1-2 negativa, 3 neutra, 4-5 positiva)")
    ap.add_argument("--sem-reescrita", action="store_true",
                    help="nao usa o LLM para reescrever a pergunta nem checar o escopo")
    ap.add_argument("--modelo", default=MODELO_LLM, help="modelo do Ollama")
    ap.add_argument("--debug", action="store_true", help="mostra etapas, contagens e tempos")
    args = ap.parse_args()

    rag = RAG(modelo_llm=args.modelo)
    reescrever = not args.sem_reescrita

    if args.pergunta:
        imprimir(rag.perguntar(" ".join(args.pergunta), args.faixa, reescrever), args.debug)
        return
    while True:
        p = input("\nPergunta (vazio para sair): ").strip()
        if not p:
            break
        imprimir(rag.perguntar(p, args.faixa, reescrever), args.debug)


if __name__ == "__main__":
    main()
