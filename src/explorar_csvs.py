"""Fase 4 (passo 2) - Conferir as colunas reais dos CSVs da Olist antes de usa-las.

Lista todos os CSVs de data/ e, para cada um, mostra tamanho, colunas, tipos,
quantidade de ausentes e as 3 primeiras linhas (head).

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo):
    python src/explorar_csvs.py
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 40)


def main() -> None:
    arquivos = sorted(PASTA.glob("*.csv"))
    if not arquivos:
        print(f"Nenhum CSV encontrado em {PASTA}")
        return
    print("CSVs encontrados:", ", ".join(a.name for a in arquivos))
    for arq in arquivos:
        df = pd.read_csv(arq)
        print("\n" + "=" * 90)
        print(f"{arq.name}  |  {df.shape[0]} linhas x {df.shape[1]} colunas")
        print("-" * 90)
        resumo = pd.DataFrame({"tipo": df.dtypes.astype(str), "ausentes": df.isna().sum()})
        print(resumo)
        print("\nhead(3):")
        print(df.head(3))


if __name__ == "__main__":
    main()
