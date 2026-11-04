"""Fase 2/4 - Geracao de embeddings e indexacao no ChromaDB.

Le data/processed/reviews_enriquecidas.csv (gerado por enriquecer_metadados.py), gera um
embedding por avaliacao e grava tudo em data/chroma_db (colecao com distancia de cosseno),
com os metadados: nota, faixa_nota, ano_mes, data_avaliacao, order_id, categoria,
status_pedido, ano_mes_compra, faixa_entrega, n_categorias e dias_atraso (so se houve entrega).

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/indexar.py

Os embeddings ficam em cache (.npy). Se mudar o CSV ou o modelo, apague o cache.
"""
import time

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    CACHE_EMBEDDINGS,
    CHROMA_DIR,
    COLECAO,
    CSV_ENRIQUECIDO,
    MODELO_EMBEDDINGS,
    PREFIXO_DOCUMENTO,
)

TAMANHO_LOTE_EMBEDDING = 64
TAMANHO_LOTE_CHROMA = 2000


def escolher_dispositivo() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def gerar_embeddings(documentos: list[str]) -> np.ndarray:
    if CACHE_EMBEDDINGS.exists():
        emb = np.load(CACHE_EMBEDDINGS)
        if emb.shape[0] == len(documentos):
            print(f"Embeddings carregados do cache: {emb.shape}")
            return emb
        print("Cache com tamanho diferente do CSV; gerando de novo.")

    dispositivo = escolher_dispositivo()
    print(f"Carregando modelo {MODELO_EMBEDDINGS} no dispositivo: {dispositivo}")
    modelo = SentenceTransformer(MODELO_EMBEDDINGS, device=dispositivo)

    textos = [PREFIXO_DOCUMENTO + d for d in documentos]
    inicio = time.time()
    emb = modelo.encode(
        textos,
        batch_size=TAMANHO_LOTE_EMBEDDING,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Embeddings gerados em {time.time() - inicio:.0f}s: {emb.shape}")
    CACHE_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_EMBEDDINGS, emb.astype("float32"))
    return emb


def montar_metadados(df: pd.DataFrame) -> list[dict]:
    """Um dicionario por avaliacao. O Chroma nao aceita valores vazios (None/NaN):
    por isso dias_atraso so entra quando o pedido tem data de entrega."""
    metadados = []
    for r in df.itertuples(index=False):
        m = {
            "nota": int(r.nota),
            "faixa_nota": r.faixa_nota,
            "ano_mes": r.ano_mes,
            "data_avaliacao": r.data_avaliacao,
            "order_id": str(r.order_id),
            "categoria": str(r.categoria),
            "status_pedido": str(r.status_pedido),
            "ano_mes_compra": str(r.ano_mes_compra),
            "faixa_entrega": str(r.faixa_entrega),
            "n_categorias": int(r.n_categorias),
        }
        if pd.notna(r.dias_atraso):
            m["dias_atraso"] = int(r.dias_atraso)
        metadados.append(m)
    return metadados


def indexar(df: pd.DataFrame, emb: np.ndarray) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reindexacao limpa: remove a colecao antiga, se existir
    existentes = [c if isinstance(c, str) else c.name for c in cliente.list_collections()]
    if COLECAO in existentes:
        cliente.delete_collection(COLECAO)

    colecao = cliente.create_collection(
        COLECAO, configuration={"hnsw": {"space": "cosine"}}
    )

    ids = df["review_id"].astype(str).tolist()
    documentos = df["documento"].tolist()
    metadados = montar_metadados(df)

    for i in range(0, len(ids), TAMANHO_LOTE_CHROMA):
        j = i + TAMANHO_LOTE_CHROMA
        colecao.add(
            ids=ids[i:j],
            embeddings=emb[i:j].tolist(),
            documents=documentos[i:j],
            metadatas=metadados[i:j],
        )
        print(f"Indexados {min(j, len(ids))}/{len(ids)}")

    print(f"\nColecao '{COLECAO}' pronta com {colecao.count()} documentos em {CHROMA_DIR}")
    for filtro in ({"faixa_entrega": "atrasada"}, {"faixa_nota": "negativa"},
                   {"categoria": "beleza_saude"}):
        n = len(colecao.get(where=filtro, include=[])["ids"])
        print(f"  conferencia do filtro {filtro}: {n} documentos")


def main() -> None:
    if not CSV_ENRIQUECIDO.exists():
        raise SystemExit(f"Nao achei {CSV_ENRIQUECIDO}. Rode antes: python src/enriquecer_metadados.py")
    df = pd.read_csv(CSV_ENRIQUECIDO)
    df["titulo"] = df["titulo"].fillna("")
    print(f"Documentos a indexar: {len(df)}")
    emb = gerar_embeddings(df["documento"].tolist())
    indexar(df, emb)


if __name__ == "__main__":
    main()
