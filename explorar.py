import pandas as pd

pd.set_option("display.max_colwidth", 200)

df = pd.read_csv("data/olist_order_reviews_dataset.csv")

print("Formato:", df.shape)
print("\nProporção de nulos por coluna:")
print(df.isna().mean())

print("\nDistribuição das notas:")
print(df["review_score"].value_counts().sort_index())

com_texto = df.dropna(subset=["review_comment_message"])
print("\nAvaliações com comentário:", len(com_texto))

tamanhos = com_texto["review_comment_message"].str.len()
print("\nTamanho dos comentários (caracteres):")
print(tamanhos.describe())

print("\nDuplicatas de review_id:", df["review_id"].duplicated().sum())
print("Comentários duplicados:", com_texto["review_comment_message"].duplicated().sum())

print("\nNotas entre os comentários:")
print(com_texto["review_score"].value_counts().sort_index())

print("\nAmostra de 10 comentários:")
for _, linha in com_texto.sample(10, random_state=42).iterrows():
    print(f"[nota {linha['review_score']}] {linha['review_comment_message']!r}")