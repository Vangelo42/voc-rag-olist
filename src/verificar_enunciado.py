"""Verificacao final - as perguntas do proprio enunciado do Tech Challenge.

Roda, uma vez cada, as perguntas de exemplo que aparecem no PDF do desafio (texto literal) e
dois casos de evidencia insuficiente, e grava avaliacao/enunciado/resultados.csv com o status,
os filtros, a saida do classificador, as citacoes e a resposta completa.

Nao faz parte das rodadas de avaliacao (avaliar.py): e uma conferencia de que o sistema
responde ao que o enunciado pede. Nada no sistema foi ajustado olhando para este resultado.

Uso (a partir da pasta do projeto, com o ambiente virtual ativo e o Ollama aberto):
    python src/verificar_enunciado.py
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "avaliacao" / "enunciado"
ARQ = PASTA / "resultados.csv"

# (id, secao do enunciado, pergunta literal, filtros manuais ou None)
PERGUNTAS = [
    ("P01", "Introducao", "Quais são os principais motivos de insatisfação dos clientes?", None),
    ("P02", "Introducao", "Quais problemas aparecem com maior frequência nas avaliações?", None),
    ("P03", "Introducao", "O que os consumidores mais valorizam em sua experiência?", None),
    ("P04", "Introducao", "Quais padrões podem ser identificados em relatos relacionados à entrega, "
                          "qualidade dos produtos ou atendimento?", None),
    ("P05", "Introducao", "Quais evidências presentes nos comentários sustentam determinado insight?", None),
    ("P06", "Problema de negocio", "Quais são os principais problemas relatados pelos clientes?", None),
    ("P07", "Problema de negocio", "Quais reclamações relacionadas à entrega aparecem com maior frequência?", None),
    ("P08", "Problema de negocio", "O que os clientes mais elogiam em suas experiências de compra?", None),
    ("P09", "Problema de negocio", "Quais problemas são relatados por clientes insatisfeitos?", None),
    ("P10", "Problema de negocio", "Existem padrões recorrentes nas avaliações relacionadas à qualidade "
                                   "dos produtos?", None),
    ("P11", "Exemplos de exploracao", "Quais são os principais problemas relatados pelos clientes em suas "
                                      "avaliações?", None),
    ("P12", "Exemplos de exploracao", "O que os clientes relatam sobre problemas relacionados à entrega?", None),
    ("P13", "Exemplos de exploracao", "Quais aspectos da experiência de compra são mais elogiados pelos "
                                      "clientes?", None),
    ("P14", "Exemplos de exploracao", "Quais padrões podem ser identificados nas avaliações de clientes "
                                      "insatisfeitos?", None),
    ("P15", "Exemplos de exploracao", "Quais avaliações sustentam a conclusão de que existem problemas "
                                      "recorrentes relacionados ao prazo de entrega?", None),
    ("P16", "Metadados", "Quais são os principais problemas relatados nas avaliações com nota baixa?", None),
    ("P17", "Metadados", "O que os clientes mais elogiam em determinadas categorias de produtos?", None),
    ("P18", "Metadados", "Quais temas aparecem nas avaliações relacionadas a experiências de entrega "
                         "insatisfatórias?", None),
    # Requisito 4: situacoes sem avaliacoes relevantes ou com evidencia insuficiente.
    # O esperado aqui e sem_evidencia (nao sao perguntas do enunciado).
    ("S01", "Requisito 4 (periodo fora da base)", "Quais problemas os clientes relataram em 2019?", None),
    ("S02", "Requisito 4 (categoria com poucas avaliacoes)", "O que os clientes elogiam?",
     {"categoria": "seguros_e_servicos"}),
]


def main() -> None:
    from rag import RAG  # import tardio: carrega os modelos

    rag = RAG()
    print("Aquecendo o modelo (esta chamada nao entra no resultado)...")
    rag.perguntar("O que os clientes elogiam?")

    PASTA.mkdir(parents=True, exist_ok=True)
    linhas = []
    for id_, secao, pergunta, manuais in PERGUNTAS:
        res = rag.perguntar(pergunta, manuais)
        d = res["detalhes"]
        saida = (d.get("saida_reescrita") or "").splitlines()
        linhas.append({
            "id": id_, "secao": secao, "pergunta": pergunta,
            "filtros_manuais": str(manuais or ""), "status": res["status"],
            "filtros": res["filtros_descricao"],
            "classificador": saida[0] if saida else "",
            "melhor_similaridade": d.get("melhor_similaridade", ""),
            "textos_unicos_acima_limiar": d.get("textos_unicos_acima_do_limiar", ""),
            "evidencias": d.get("evidencias_selecionadas", 0),
            "citacoes_validas": ";".join(map(str, res["citacoes_validas"])),
            "freq_detectada": ";".join(d.get("frequencia_detectada", [])),
            "freq_apos_reescrita": ";".join(d.get("frequencia_apos_reescrita", [])),
            "avisos": " | ".join(res["avisos"]),
            "t_total": d.get("tempos_s", {}).get("total", ""),
            "resposta": res["resposta"],
        })
        pd.DataFrame(linhas).to_csv(ARQ, index=False, encoding="utf-8-sig")
        print(f"\n[{id_}] {pergunta}")
        print(f"     status: {res['status']} | filtros: {res['filtros_descricao']} | "
              f"classificador: {linhas[-1]['classificador']}")

    df = pd.DataFrame(linhas)
    enun = df[df["id"].str.startswith("P")]
    print("\n" + "=" * 72)
    print(f"Perguntas do enunciado respondidas: {(enun['status'] == 'respondida').sum()}/{len(enun)}")
    for _, r in enun[enun["status"] != "respondida"].iterrows():
        print(f"   NAO respondida: {r['id']} ({r['status']}; {r['classificador']}): {r['pergunta']}")
    for _, r in df[df["id"].str.startswith("S")].iterrows():
        print(f"{r['id']} ({r['secao']}): {r['status']} | filtros: {r['filtros']}")
    print(f"\nResultados e respostas completas: {ARQ}")


if __name__ == "__main__":
    main()
