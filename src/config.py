"""Configuracoes compartilhadas do projeto (caminhos, modelo de embeddings, prefixos)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

CSV_PROCESSADO = RAIZ / "data" / "processed" / "reviews_processadas.csv"
CSV_ENRIQUECIDO = RAIZ / "data" / "processed" / "reviews_enriquecidas.csv"  # gerado por enriquecer_metadados.py
CACHE_EMBEDDINGS = RAIZ / "data" / "processed" / "embeddings_e5_base.npy"
CHROMA_DIR = RAIZ / "data" / "chroma_db"
COLECAO = "avaliacoes_olist"

# Modelo multilingue de embeddings (funciona bem em portugues, roda local)
MODELO_EMBEDDINGS = "intfloat/multilingual-e5-base"

# O familia E5 foi treinada com prefixos: um para os documentos, outro para as perguntas.
# Sem eles a qualidade da busca cai.
PREFIXO_DOCUMENTO = "passage: "
PREFIXO_PERGUNTA = "query: "
