"""Fase 3 - Pipeline RAG (Voice of Customer sobre avaliacoes da Olist).

Fluxo:
  pergunta -> processamento (LLM reescreve e checa escopo) -> embeddings
  -> recuperacao no ChromaDB -> limiar de relevancia + deduplicacao
  -> contexto numerado -> geracao (Ollama) -> checagem de citacoes

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo e o Ollama aberto):
    python src/rag.py "Quais os principais problemas de entrega?"
    python src/rag.py "O que os clientes elogiam?" --faixa positiva --debug
    python src/rag.py                      # modo interativo
"""
import argparse
import os
import re
import unicodedata

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLECAO, MODELO_EMBEDDINGS, PREFIXO_PERGUNTA

# --- Parametros do pipeline (justificar no README) -------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = "qwen3:8b"
LIMIAR = 0.85          # similaridade minima para uma avaliacao contar como evidencia
MIN_EVIDENCIAS = 3     # evidencias unicas minimas para tentar responder
TOP_K_BUSCA = 30       # candidatos por consulta (busca larga, depois filtra e deduplica)
MAX_EVIDENCIAS = 10    # evidencias enviadas ao modelo
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
1. Use apenas o que está nas avaliações. Não use conhecimento externo e não invente fatos.
2. Cite as avaliações que sustentam cada afirmação com o número entre colchetes, por exemplo [2][5].
3. Não informe percentuais, totais nem expressões como "a maioria dos clientes": as avaliações \
são uma amostra recuperada, não uma estatística.
4. Se as avaliações não responderem à pergunta, responda exatamente: \
Não encontrei evidências suficientes nas avaliações para responder a essa pergunta.
5. Responda em português do Brasil, em no máximo 8 linhas: uma frase de síntese e depois tópicos curtos por tema."""


# --- Funcoes auxiliares ------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minusculas, sem acentos nem pontuacao: usado para achar quase duplicatas."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def deduplicar(itens: list[dict], maximo: int) -> list[dict]:
    """Mantem a primeira ocorrencia de cada texto (itens ja ordenados por similaridade)."""
    vistos, unicos = set(), []
    for item in itens:
        chave = normalizar(item["doc"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(item)
        if len(unicos) == maximo:
            break
    return unicos


def chamar_ollama(mensagens: list[dict], modelo: str, temperatura: float = 0.1) -> str:
    corpo = {
        "model": modelo,
        "messages": mensagens,
        "stream": False,
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

    # Etapa 2: recuperacao semantica
    def buscar(self, consultas: list[str], faixa: str | None = None) -> list[dict]:
        onde = {"faixa_nota": faixa} if faixa else None
        achados: dict[str, dict] = {}
        for texto in consultas:
            vetor = self.encoder.encode(PREFIXO_PERGUNTA + texto, normalize_embeddings=True)
            res = self.colecao.query(
                query_embeddings=[vetor.tolist()],
                n_results=TOP_K_BUSCA,
                where=onde,
                include=["documents", "metadatas", "distances"],
            )
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                sim = 1 - dist
                if id_ not in achados or sim > achados[id_]["sim"]:
                    achados[id_] = {"id": id_, "doc": doc, "meta": meta, "sim": sim}
        return sorted(achados.values(), key=lambda x: -x["sim"])

    def perguntar(self, pergunta: str, faixa: str | None = None, reescrever: bool = True) -> dict:
        resultado = {
            "pergunta": pergunta, "status": None, "resposta": None, "evidencias": [],
            "citacoes_validas": [], "citacoes_invalidas": [], "avisos": [], "detalhes": {},
        }

        consultas, fora = self.processar_pergunta(pergunta, reescrever)
        resultado["detalhes"]["consultas"] = consultas
        if fora:
            resultado.update(status="fora_do_escopo", resposta=MSG_FORA_ESCOPO)
            return resultado

        candidatos = self.buscar(consultas, faixa)
        relevantes = [c for c in candidatos if c["sim"] >= self.limiar]
        evidencias = deduplicar(relevantes, MAX_EVIDENCIAS)
        resultado["detalhes"].update(
            candidatos=len(candidatos),
            acima_do_limiar=len(relevantes),
            evidencias_unicas=len(evidencias),
            melhor_similaridade=round(candidatos[0]["sim"], 3) if candidatos else None,
        )

        if len(evidencias) < self.min_evidencias:
            resultado.update(status="sem_evidencia", resposta=MSG_SEM_EVIDENCIA)
            return resultado

        mensagens = [
            {"role": "system", "content": SISTEMA_RESPOSTA},
            {"role": "user",
             "content": f"PERGUNTA: {pergunta}\n\nAVALIAÇÕES:\n{montar_contexto(evidencias)}"},
        ]
        resposta = chamar_ollama(mensagens, self.modelo_llm)

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
    ap.add_argument("--debug", action="store_true", help="mostra etapas e contagens")
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
