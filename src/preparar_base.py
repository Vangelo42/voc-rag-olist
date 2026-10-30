"""Fase 1 - Preparacao da base de conhecimento (avaliacoes Olist).

Le olist_order_reviews_dataset.csv, aplica selecao e limpeza e grava
data/processed/reviews_processadas.csv, com um documento por avaliacao.

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/preparar_base.py
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "olist_order_reviews_dataset.csv"
SAIDA = RAIZ / "data" / "processed" / "reviews_processadas.csv"

# Parametros ajustaveis (justificar no README)
MIN_PALAVRAS = 2   # comentarios com menos palavras sao pouco informativos
MIN_LETRAS = 3     # descarta comentarios sem conteudo textual (".", "!!!", "...")


def limpar_texto(valor) -> str:
    """Remove quebras de linha e espacos repetidos, sem alterar acentos nem caixa."""
    texto = str(valor).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", texto).strip()


def faixa_da_nota(nota: int) -> str:
    if nota <= 2:
        return "negativa"
    if nota == 3:
        return "neutra"
    return "positiva"


def montar_documento(titulo: str, texto: str) -> str:
    """Junta titulo e comentario quando o titulo existe."""
    if not titulo:
        return texto
    return f"{titulo.rstrip('.!?')}. {texto}"


def main() -> None:
    etapas = []  # (descricao, quantidade restante)

    df = pd.read_csv(ENTRADA)
    etapas.append(("Avaliacoes na base original", len(df)))

    # 1. Selecao: so entram avaliacoes que tem comentario escrito
    df = df.dropna(subset=["review_comment_message"]).copy()
    etapas.append(("Com comentario escrito", len(df)))

    # 2. Limpeza basica de texto e tratamento do titulo ausente
    df["texto"] = df["review_comment_message"].map(limpar_texto)
    df["titulo"] = df["review_comment_title"].fillna("").map(limpar_texto)
    df = df[df["texto"] != ""]
    etapas.append(("Apos remover textos vazios", len(df)))

    # 3. Duplicatas de review_id (mesma avaliacao repetida)
    df = df.drop_duplicates(subset="review_id", keep="first")
    etapas.append(("Apos remover review_id duplicado", len(df)))

    # 4. Filtro de conteudo minimo
    n_palavras = df["texto"].str.split().str.len()
    n_letras = df["texto"].str.count(r"[A-Za-z\u00c0-\u00ff]")
    df = df[(n_palavras >= MIN_PALAVRAS) & (n_letras >= MIN_LETRAS)]
    etapas.append((f"Apos filtro minimo ({MIN_PALAVRAS}+ palavras)", len(df)))

    # 5. Documento e metadados
    df["documento"] = [montar_documento(t, x) for t, x in zip(df["titulo"], df["texto"])]
    df["nota"] = df["review_score"].astype(int)
    df["faixa_nota"] = df["nota"].map(faixa_da_nota)
    data = pd.to_datetime(df["review_creation_date"])
    df["data_avaliacao"] = data.dt.strftime("%Y-%m-%d")
    df["ano_mes"] = data.dt.strftime("%Y-%m")
    # Versao normalizada, usada depois para remover quase duplicatas na recuperacao
    df["texto_normalizado"] = (
        df["documento"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    )

    saida = df[
        [
            "review_id", "order_id", "nota", "faixa_nota", "data_avaliacao",
            "ano_mes", "titulo", "texto", "documento", "texto_normalizado",
        ]
    ].reset_index(drop=True)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(SAIDA, index=False, encoding="utf-8")

    # Relatorio (use estes numeros no README)
    print("=== Funil de preparacao ===")
    for descricao, qtd in etapas:
        print(f"{descricao:<45} {qtd:>7}")
    print(f"\nArquivo salvo em: {SAIDA}")
    print("\n=== Documentos por nota ===")
    print(saida["nota"].value_counts().sort_index())
    print("\n=== Periodo ===")
    print(saida["data_avaliacao"].min(), "ate", saida["data_avaliacao"].max())
    print("\n=== Tamanho dos documentos (caracteres) ===")
    print(saida["documento"].str.len().describe())
    print("\nTextos repetidos (mantidos de proposito):",
          saida["texto_normalizado"].duplicated().sum())
    print("\n=== Amostra ===")
    for _, linha in saida.sample(min(8, len(saida)), random_state=1).iterrows():
        print(f"[nota {linha['nota']}] {linha['documento']}")


if __name__ == "__main__":
    main()
