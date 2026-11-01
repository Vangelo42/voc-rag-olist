"""Fase 3 (preparacao) - Calibracao do limiar de relevancia.

Roda perguntas DENTRO e FORA do escopo da base e mostra como a similaridade se
distribui, para escolher (1) o limiar de similaridade e (2) o minimo de avaliacoes
acima do limiar exigido para responder.

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/calibrar_limiar.py

Saida tambem gravada em data/processed/calibracao.csv (use no README).
"""
import csv

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLECAO, MODELO_EMBEDDINGS, PREFIXO_PERGUNTA, RAIZ

TOP_K = 10
LIMIARES = [0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]
MIN_DOCS = [1, 3, 5]

DENTRO = [
    "entrega atrasada",
    "produto chegou quebrado",
    "o que os clientes elogiam",
    "atendimento ruim, ninguem responde",
    "quais os principais problemas relatados pelos clientes",
    "produto diferente do anunciado",
    "nao recebi o produto",
    "qualidade do produto",
    "embalagem danificada",
    "entrega rapida e antes do prazo",
    "vendedor nao respondeu minhas mensagens",
    "produto com defeito",
    "troca ou reembolso nao resolvido",
    "faltou item no pedido",
    "o que os clientes mais valorizam na compra",
    "clientes insatisfeitos com o prazo de entrega",
]

FORA = [
    "qual a previsao do tempo para amanha?",
    "receita de bolo de cenoura",
    "quem ganhou a copa do mundo de 2018?",
    "como instalar python no windows",
    "qual a capital da franca",
    "qual o preco do dolar hoje",
    "me conte uma piada",
    "como funciona a bolsa de valores",
    "quais sao os melhores filmes de 2020",
    "como tratar dor de cabeca",
    "qual o salario medio de um programador",
    "resuma a historia do brasil",
    # Dificeis: parecem do dominio, mas as avaliacoes nao respondem
    "quantos pedidos a empresa teve em 2017?",
    "qual o telefone do suporte da loja?",
]


def similaridades(modelo, colecao, pergunta: str) -> np.ndarray:
    vetor = modelo.encode(PREFIXO_PERGUNTA + pergunta, normalize_embeddings=True)
    res = colecao.query(
        query_embeddings=[vetor.tolist()], n_results=TOP_K, include=["distances"]
    )
    return np.array([1 - d for d in res["distances"][0]])


def main() -> None:
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    colecao = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLECAO)

    resultados = {"dentro": {}, "fora": {}}
    for grupo, perguntas in (("dentro", DENTRO), ("fora", FORA)):
        for p in perguntas:
            resultados[grupo][p] = similaridades(modelo, colecao, p)

    linhas = []
    for grupo in ("dentro", "fora"):
        print(f"\n=== Perguntas {grupo.upper()} do escopo (ordenadas pelo top-1) ===")
        print(f"{'top1':>6} {'top5 med':>9} {'5o':>6}  pergunta")
        ordenado = sorted(resultados[grupo].items(), key=lambda kv: -kv[1][0])
        for p, s in ordenado:
            print(f"{s[0]:6.3f} {s[:5].mean():9.3f} {s[4]:6.3f}  {p}")
            linhas.append([grupo, p, round(s[0], 4), round(s[:5].mean(), 4), round(s[4], 4)])

    print("\n=== Resumo do top-1 ===")
    for grupo in ("dentro", "fora"):
        t1 = np.array([s[0] for s in resultados[grupo].values()])
        print(
            f"{grupo:>6}: min={t1.min():.3f}  mediana={np.median(t1):.3f}  max={t1.max():.3f}"
        )

    print("\n=== Regra: responder se houver >= N avaliacoes com similaridade >= limiar ===")
    print("  (aceitas = % das perguntas DENTRO respondidas | rejeitadas = % das FORA recusadas)")
    print(f"{'limiar':>7} {'N':>3} {'aceitas':>8} {'rejeitadas':>11}")
    grade = []
    for limiar in LIMIARES:
        for n in MIN_DOCS:
            aceitas = np.mean([(s >= limiar).sum() >= n for s in resultados["dentro"].values()])
            rejeitadas = np.mean([(s >= limiar).sum() < n for s in resultados["fora"].values()])
            print(f"{limiar:7.2f} {n:3d} {aceitas:8.0%} {rejeitadas:11.0%}")
            grade.append([limiar, n, round(aceitas, 3), round(rejeitadas, 3)])

    saida = RAIZ / "data" / "processed" / "calibracao.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "pergunta", "top1", "media_top5", "quinto"])
        w.writerows(linhas)
        w.writerow([])
        w.writerow(["limiar", "min_docs", "aceitas_dentro", "rejeitadas_fora"])
        w.writerows(grade)
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
