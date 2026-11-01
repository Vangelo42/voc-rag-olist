"""Fase 2 - Busca manual para validar a recuperacao semantica.

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/buscar_teste.py                       # roda perguntas de exemplo
    python src/buscar_teste.py "produto veio quebrado"   # sua propria pergunta

Mostra a similaridade (1 - distancia de cosseno) de cada resultado. Essas notas
servirao para calibrar o limiar de relevancia na Fase 3.
"""
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLECAO, MODELO_EMBEDDINGS, PREFIXO_PERGUNTA

PERGUNTAS_EXEMPLO = [
    "entrega atrasada",
    "produto chegou quebrado",
    "o que os clientes elogiam",
    "atendimento ruim, ninguem responde",
    # Fora do escopo de proposito: para ver como a similaridade se comporta
    "qual a previsao do tempo para amanha?",
    "receita de bolo de cenoura",
]
TOP_K = 5


def main() -> None:
    perguntas = sys.argv[1:] or PERGUNTAS_EXEMPLO
    modelo = SentenceTransformer(MODELO_EMBEDDINGS)
    colecao = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLECAO)

    for pergunta in perguntas:
        vetor = modelo.encode(PREFIXO_PERGUNTA + pergunta, normalize_embeddings=True)
        res = colecao.query(
            query_embeddings=[vetor.tolist()],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\n=== {pergunta} ===")
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            print(f"  sim={1 - dist:.3f} | nota {meta['nota']} | {doc}")


if __name__ == "__main__":
    main()
