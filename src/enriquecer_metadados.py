"""Fase 4 (passo 3) - Enriquecimento dos metadados das avaliacoes.

Le data/processed/reviews_processadas.csv e acrescenta, por pedido:
  - categoria / categoria_en : categoria do produto (order_items -> products -> traducao)
  - n_categorias             : quantas categorias diferentes o pedido tem
  - status_pedido            : order_status
  - ano_mes_compra           : mes da compra (order_purchase_timestamp)
  - dias_atraso              : data de entrega ao cliente - data estimada (em dias; negativo = adiantado)
  - faixa_entrega            : no_prazo | atrasada | nao_entregue

Grava data/processed/reviews_enriquecidas.csv com as MESMAS linhas, na MESMA ordem do arquivo
de entrada (por isso o cache de embeddings continua valido).

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/enriquecer_metadados.py
"""
from pathlib import Path

import pandas as pd

from config import CSV_PROCESSADO, RAIZ

DADOS = RAIZ / "data"
SAIDA = RAIZ / "data" / "processed" / "reviews_enriquecidas.csv"

SEM_CATEGORIA = "sem_categoria"


def carregar_pedidos() -> pd.DataFrame:
    """Uma linha por pedido, com status, mes da compra e atraso na entrega."""
    cols = [
        "order_id", "order_status", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date",
    ]
    ped = pd.read_csv(DADOS / "olist_orders_dataset.csv", usecols=cols)
    assert ped["order_id"].is_unique, "order_id repetido em olist_orders_dataset.csv"

    compra = pd.to_datetime(ped["order_purchase_timestamp"], errors="coerce")
    entregue = pd.to_datetime(ped["order_delivered_customer_date"], errors="coerce")
    estimada = pd.to_datetime(ped["order_estimated_delivery_date"], errors="coerce")

    # Compara so as DATAS: a estimada vem as 00:00; entregar no proprio dia estimado nao e atraso.
    dias = (entregue.dt.normalize() - estimada.dt.normalize()).dt.days
    faixa = pd.Series("no_prazo", index=ped.index)
    faixa[dias > 0] = "atrasada"
    faixa[entregue.isna()] = "nao_entregue"

    return pd.DataFrame({
        "order_id": ped["order_id"],
        "status_pedido": ped["order_status"],
        "ano_mes_compra": compra.dt.strftime("%Y-%m"),
        "dias_atraso": dias,
        "faixa_entrega": faixa,
    })


def carregar_categorias() -> pd.DataFrame:
    """Uma categoria por pedido. Pedido com varios itens: vale a categoria mais frequente
    (empate: a do primeiro item); 'sem_categoria' so vence se for a unica."""
    itens = pd.read_csv(DADOS / "olist_order_items_dataset.csv",
                        usecols=["order_id", "order_item_id", "product_id"])
    prod = pd.read_csv(DADOS / "olist_products_dataset.csv",
                       usecols=["product_id", "product_category_name"])
    trad = pd.read_csv(DADOS / "product_category_name_translation.csv")

    itens = itens.merge(prod, on="product_id", how="left")
    itens["categoria"] = itens["product_category_name"].fillna(SEM_CATEGORIA)

    por_cat = (
        itens.groupby(["order_id", "categoria"])
        .agg(n=("order_item_id", "size"), primeiro=("order_item_id", "min"))
        .reset_index()
    )
    por_cat["sem"] = por_cat["categoria"] == SEM_CATEGORIA
    por_cat = por_cat.sort_values(["order_id", "sem", "n", "primeiro"],
                                  ascending=[True, True, False, True])
    escolhida = por_cat.drop_duplicates("order_id")[["order_id", "categoria"]]
    escolhida = escolhida.merge(
        por_cat.groupby("order_id")["categoria"].nunique().rename("n_categorias").reset_index(),
        on="order_id",
    )

    mapa = dict(zip(trad["product_category_name"], trad["product_category_name_english"]))
    escolhida["categoria_en"] = escolhida["categoria"].map(mapa).fillna(escolhida["categoria"])
    return escolhida


def main() -> None:
    rev = pd.read_csv(CSV_PROCESSADO)
    n_entrada = len(rev)
    ordem = rev["review_id"].tolist()

    pedidos = carregar_pedidos()
    categorias = carregar_categorias()

    df = rev.merge(pedidos, on="order_id", how="left").merge(categorias, on="order_id", how="left")

    # Garantias: mesmas linhas, na mesma ordem (o cache de embeddings depende disso)
    assert len(df) == n_entrada, f"o merge mudou o numero de linhas: {n_entrada} -> {len(df)}"
    assert df["review_id"].tolist() == ordem, "a ordem das linhas mudou"

    df["status_pedido"] = df["status_pedido"].fillna("desconhecido")
    df["faixa_entrega"] = df["faixa_entrega"].fillna("nao_entregue")
    df["ano_mes_compra"] = df["ano_mes_compra"].fillna("desconhecido")
    df["categoria"] = df["categoria"].fillna(SEM_CATEGORIA)
    df["categoria_en"] = df["categoria_en"].fillna(SEM_CATEGORIA)
    df["n_categorias"] = df["n_categorias"].fillna(0).astype(int)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False, encoding="utf-8")

    # --- Relatorio (use estes numeros no README) ---------------------------------
    print(f"Linhas: {len(df)} (entrada: {n_entrada}) | arquivo: {SAIDA}")
    print("\n=== Status do pedido ===")
    print(df["status_pedido"].value_counts())
    print("\n=== Entrega (faixa_entrega) ===")
    print(df["faixa_entrega"].value_counts())
    print("\n=== dias_atraso (so pedidos entregues) ===")
    print(df["dias_atraso"].describe())
    print("\n=== Pedidos com mais de uma categoria ===")
    print(f"{(df['n_categorias'] > 1).sum()} de {len(df)} ({(df['n_categorias'] > 1).mean():.1%})")
    print(f"Sem categoria: {(df['categoria'] == SEM_CATEGORIA).sum()}")
    print("\n=== Top 15 categorias ===")
    print(df["categoria"].value_counts().head(15))
    print(f"\nCategorias distintas: {df['categoria'].nunique()}")
    sem_trad = df.loc[df["categoria"] == df["categoria_en"], "categoria"].unique()
    print("Sem traducao (ficam com o nome original):",
          [c for c in sem_trad if c != SEM_CATEGORIA])
    print("\n=== Por que o metadado de entrega ajuda: faixa de nota por faixa de entrega ===")
    print((pd.crosstab(df["faixa_entrega"], df["faixa_nota"], normalize="index") * 100).round(1))
    print("\n=== Amostra ===")
    print(df[["nota", "faixa_entrega", "dias_atraso", "categoria", "ano_mes_compra",
              "documento"]].sample(min(6, len(df)), random_state=1).to_string(max_colwidth=50))


if __name__ == "__main__":
    main()
