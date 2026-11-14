"""Fase 5 - Avaliacao do sistema (comportamento, tempos e precisao das citacoes).

Tres modos, nesta ordem:
  1) --executar : roda as perguntas de teste e grava avaliacao/resultados.csv
                  e avaliacao/citacoes_para_rotular.csv
  2) --rotular  : voce le cada afirmacao e a avaliacao citada e marca se sustenta (s/p/n);
                  o progresso e salvo a cada resposta e pode ser retomado
  3) --resumo   : calcula os numeros do README e grava avaliacao/resumo.txt
  (opcional) --revisar [--rotulo nao]: relê os pares com um rótulo (padrão: 'nao') e deixa
                  você mudar a decisão; útil para conferir o critério antes do resumo final

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo e o Ollama aberto):
    python src/avaliar.py --executar
    python src/avaliar.py --rotular
    python src/avaliar.py --resumo
Rodadas: sem --rodada os arquivos ficam em avaliacao/ (rodada 1); com --rodada rodada2 ficam em
avaliacao/rodada2/ (use depois de mudar o sistema, para comparar antes e depois).
A lista cresceu a cada rodada: 23 perguntas na 1, 32 na 2 e 40 na 3.
"""
import argparse
import random
import re
import statistics
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "avaliacao"
ARQ_RESULTADOS = PASTA / "resultados.csv"
ARQ_CITACOES = PASTA / "citacoes_para_rotular.csv"
ARQ_RESUMO = PASTA / "resumo.txt"


def definir_rodada(rodada: str) -> None:
    """Escolhe a pasta dos arquivos: avaliacao/ (rodada 1) ou avaliacao/<rodada>/."""
    global PASTA, ARQ_RESULTADOS, ARQ_CITACOES, ARQ_RESUMO
    PASTA = RAIZ / "avaliacao" / rodada if rodada else RAIZ / "avaliacao"
    ARQ_RESULTADOS = PASTA / "resultados.csv"
    ARQ_CITACOES = PASTA / "citacoes_para_rotular.csv"
    ARQ_RESUMO = PASTA / "resumo.txt"

REPETICOES_FORA = 3  # o LLM varia: perguntas fora do escopo rodam mais de uma vez

# (id, tipo, pergunta)
#   valida      : deve ser respondida
#   fora_escopo : deve ser recusada (fora_do_escopo ou sem_evidencia; ambas contam como recusa)
#   dificil     : so registrada (resultado esperado incerto); avaliar o texto manualmente
PERGUNTAS = [
    ("V01", "valida", "Quais são os principais motivos de insatisfação dos clientes?"),
    ("V02", "valida", "O que os consumidores mais valorizam em sua experiência de compra?"),
    ("V03", "valida", "O que os clientes relatam sobre a qualidade dos produtos?"),
    ("V04", "valida", "Existem relatos de produtos que chegaram diferentes do que foi anunciado?"),
    ("V05", "valida", "Quais padrões aparecem nas avaliações de clientes insatisfeitos?"),
    ("V06", "valida", "Quais avaliações sustentam a conclusão de que existem problemas "
                      "recorrentes com o prazo de entrega?"),
    ("V07", "valida", "Quais problemas de entrega aparecem nas avaliações de 2018?"),
    ("V08", "valida", "O que os clientes elogiam em produtos de beleza e saúde?"),
    ("V09", "valida", "Quais problemas são relatados em móveis e decoração?"),
    ("V10", "valida", "Quais reclamações aparecem nas avaliações com nota 1?"),
    ("V11", "valida", "O que os clientes dizem sobre pedidos que não chegaram?"),
    ("V12", "valida", "Quais elogios aparecem nas compras de dezembro de 2017?"),
    ("V13", "valida", "Quais problemas de qualidade aparecem em informática e acessórios?"),
    ("V14", "valida", "O que os clientes relatam sobre o atendimento dos vendedores?"),
    ("V15", "valida", "Quais problemas aparecem nas avaliações de esporte e lazer?"),
    ("V16", "valida", "O que os clientes elogiam nas avaliações de 2017?"),
    ("V17", "valida", "Quais reclamações aparecem nas avaliações com nota 2?"),
    ("V18", "valida", "Quais pontos negativos aparecem nas compras de junho de 2018?"),
    ("V19", "valida", "O que as avaliações dizem sobre a embalagem dos produtos?"),
    ("V20", "valida", "Quais problemas aparecem nas avaliações de clientes que receberam "
                      "o pedido com atraso?"),
    ("V21", "valida", "Quais reclamações sobre atraso na entrega apareceram em 2017?"),
    ("V22", "valida", "Quais elogios sobre a entrega surgiram em 2018?"),
    ("V23", "valida", "Que problemas com os produtos os clientes relataram em janeiro de 2018?"),
    ("V24", "valida", "Quais críticas ao atendimento apareceram nas compras de novembro de 2017?"),
    ("V25", "valida", "O que os clientes de 2017 dizem sobre a qualidade dos produtos?"),
    ("E01", "fora_escopo", "Qual foi o faturamento da empresa em 2017?"),
    ("E02", "fora_escopo", "Quem é o CEO da Olist?"),
    ("E03", "fora_escopo", "Explique como funciona o aprendizado de máquina."),
    ("E04", "fora_escopo", "Qual o endereço da loja física?"),
    ("E05", "fora_escopo", "Qual é o melhor smartphone para comprar hoje?"),
    ("E06", "fora_escopo", "Quantas avaliações têm nota 5?"),
    ("E07", "fora_escopo", "Escreva um poema sobre a Olist."),
    ("E08", "fora_escopo", "Qual o prazo de entrega para o CEP 69000-000?"),
    ("E09", "fora_escopo", "Quantas reclamações de atraso houve em 2017?"),
    ("E10", "fora_escopo", "Qual foi o mês com mais vendas em 2018?"),
    ("E11", "fora_escopo", "Qual a nota média das avaliações em 2018?"),
    ("D01", "dificil", "O que os clientes dizem sobre o aplicativo da loja?"),
    ("D02", "dificil", "Quais elogios aparecem sobre entregas atrasadas?"),
    ("D03", "dificil", "Entrega"),
    ("D04", "dificil", "Os clientes reclamam de cobrança duplicada no cartão?"),
]

# Perguntas acrescentadas DEPOIS da rodada 1, sem ajustar o sistema a elas: mesmo tipo das que
# falharam (periodo, nota, "nas avaliacoes"), mas com outras palavras. O resumo as separa.
NOVAS = {"V15", "V16", "V17", "V18", "V19", "V20", "E06", "E07", "E08"}
# Rodada 3: V21 foi recusada por engano depois da rodada 2 (botao de exemplo da interface) e
# motivou o ajuste do classificador; as NOVAS_R3 foram escritas antes de rodar e nao foram
# usadas no ajuste (periodo + relato deve ser respondido; periodo + numero deve ser recusado).
AJUSTE_R3 = {"V21"}
NOVAS_R3 = {"V22", "V23", "V24", "V25", "E09", "E10", "E11"}


# --- Modo 1: executar -----------------------------------------------------------------
def extrair_pares(res: dict) -> list[dict]:
    """Pares (afirmacao, avaliacao citada) de uma resposta. Cada trecho termina num grupo de
    citacoes, ex.: 'produtos que nao chegaram [3][9]' gera dois pares."""
    pares = []
    total = len(res["evidencias"])
    for linha in res["resposta"].splitlines():
        if not re.search(r"\[\d+\]", linha):
            continue
        resto = linha
        while True:
            m = re.search(r"((?:\[\d+\])+)", resto)
            if not m:
                break
            trecho = re.sub(r"^[\s,;.\-–]*(e\s+)?", "", resto[:m.start()]).strip()
            if not trecho:  # citacao no inicio da frase: usa o que vem depois
                trecho = re.sub(r"\[\d+\]", "", resto[m.end():]).strip()
            if not trecho:  # a linha so lista a avaliacao como evidencia
                trecho = "(sem texto na resposta, só a citação; linha: " + linha.strip() + ")"
            for n in sorted({int(x) for x in re.findall(r"\d+", m.group(1))}):
                if 1 <= n <= total:
                    pares.append({"afirmacao": trecho, "linha_completa": linha.strip(),
                                  "n": n, "avaliacao": res["evidencias"][n - 1]["doc"],
                                  "nota": res["evidencias"][n - 1]["meta"]["nota"]})
            resto = resto[m.end():]
    return pares


def executar(repeticoes: int) -> None:
    from rag import RAG  # import tardio: so este modo precisa carregar os modelos

    rag = RAG()
    print("Aquecendo o modelo (esta chamada nao entra na avaliacao)...")
    rag.perguntar("O que os clientes elogiam?")

    PASTA.mkdir(parents=True, exist_ok=True)
    linhas, pares = [], []
    total = sum(REPETICOES_FORA if t == "fora_escopo" else repeticoes for _, t, _ in PERGUNTAS)
    feito = 0
    for id_, tipo, pergunta in PERGUNTAS:
        for rep in range(1, (REPETICOES_FORA if tipo == "fora_escopo" else repeticoes) + 1):
            feito += 1
            res = rag.perguntar(pergunta)
            d, t = res["detalhes"], res["detalhes"].get("tempos_s", {})
            if tipo == "valida":
                ok = res["status"] == "respondida"
            elif tipo == "fora_escopo":
                ok = res["status"] in ("fora_do_escopo", "sem_evidencia")
            else:
                ok = ""
            linhas.append({
                "id": id_, "tipo": tipo, "repeticao": rep, "pergunta": pergunta,
                "status": res["status"], "ok": ok, "filtros": res["filtros_descricao"],
                "melhor_similaridade": d.get("melhor_similaridade", ""),
                "textos_unicos_acima_limiar": d.get("textos_unicos_acima_do_limiar", ""),
                "evidencias": d.get("evidencias_selecionadas", 0),
                "citacoes_validas": ";".join(map(str, res["citacoes_validas"])),
                "citacoes_invalidas": ";".join(map(str, res["citacoes_invalidas"])),
                "avisos": " | ".join(res["avisos"]),
                "saida_reescrita": d.get("saida_reescrita", ""),
                "freq_detectada": ";".join(d.get("frequencia_detectada", [])),
                "freq_apos_reescrita": ";".join(d.get("frequencia_apos_reescrita", [])),
                "t_reescrita": t.get("reescrita", ""), "t_busca": t.get("busca", ""),
                "t_geracao": t.get("geracao", ""), "t_total": t.get("total", ""),
                "resposta": res["resposta"],
            })
            if tipo == "valida" and rep == 1 and res["status"] == "respondida":
                for p in extrair_pares(res):
                    pares.append({"id": id_, "pergunta": pergunta, **p, "sustenta": ""})
            print(f"[{feito}/{total}] {id_} rep {rep}: {res['status']} ({t.get('total', '?')} s)")
            pd.DataFrame(linhas).to_csv(ARQ_RESULTADOS, index=False, encoding="utf-8-sig")

    random.Random(42).shuffle(pares)  # rotular so o comeco da lista ja e uma amostra aleatoria
    pd.DataFrame(pares).to_csv(ARQ_CITACOES, index=False, encoding="utf-8-sig")
    print(f"\nResultados: {ARQ_RESULTADOS}\nCitacoes a rotular ({len(pares)}): {ARQ_CITACOES}")
    print("Proximo passo: python src/avaliar.py --rotular")


# --- Modo 2: rotular ---------------------------------------------------------------------
def rotular() -> None:
    if not ARQ_CITACOES.exists():
        raise SystemExit("Rode antes: python src/avaliar.py --executar")
    df = pd.read_csv(ARQ_CITACOES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    pendentes = df.index[df["sustenta"] == ""].tolist()
    print(f"{len(df) - len(pendentes)} rotuladas, {len(pendentes)} pendentes. "
          "Quanto mais, melhor; uns 60 ja dao uma boa amostra.\n"
          "Pergunta a cada par: a avaliacao citada SUSTENTA a afirmacao?\n"
          "  s = sim | p = parcialmente | n = nao | q = sair (o progresso e salvo)\n")
    mapa = {"s": "sim", "p": "parcial", "n": "nao"}
    for i in pendentes:
        r = df.loc[i]
        print("-" * 72)
        print(f"({len(df) - len(df.index[df['sustenta'] == '']) + 1}/{len(df)}) "
              f"Pergunta {r['id']}: {r['pergunta']}")
        print(f"Afirmação: {r['afirmacao']}")
        print(f"Avaliação [{r['n']}] (nota {r['nota']}): {r['avaliacao']}")
        while True:
            resp = input("Sustenta? (s/p/n/q): ").strip().lower()
            if resp in mapa or resp == "q":
                break
        if resp == "q":
            break
        df.loc[i, "sustenta"] = mapa[resp]
        df.to_csv(ARQ_CITACOES, index=False, encoding="utf-8-sig")
    print("\nProgresso salvo. Para ver os numeros: python src/avaliar.py --resumo")


# --- Modo 2b: revisar -----------------------------------------------------------------------
def revisar(rotulo: str) -> None:
    if not ARQ_CITACOES.exists():
        raise SystemExit("Rode antes: python src/avaliar.py --executar")
    df = pd.read_csv(ARQ_CITACOES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    alvo = df.index[df["sustenta"] == rotulo].tolist()
    if not alvo:
        raise SystemExit(f"Nenhum par com o rotulo '{rotulo}'.")
    print(f"{len(alvo)} pares com o rotulo atual '{rotulo}'.\n"
          "Criterio: s = a avaliacao, sozinha, diz o que a afirmacao diz (mesmo com outras palavras)\n"
          "          p = diz so parte, ou e do mesmo tema sem dizer exatamente isso\n"
          "          n = nao diz nada sobre o que a afirmacao diz, ou diz o contrario\n"
          "Enter = manter o rotulo | q = sair (o progresso e salvo)\n")
    mapa = {"s": "sim", "p": "parcial", "n": "nao"}
    for k, i in enumerate(alvo, start=1):
        r = df.loc[i]
        print("-" * 72)
        print(f"({k}/{len(alvo)}) Pergunta {r['id']}: {r['pergunta']}")
        print(f"Afirmação: {r['afirmacao']}")
        print(f"Avaliação [{r['n']}] (nota {r['nota']}): {r['avaliacao']}")
        print(f"Rótulo atual: {r['sustenta']}")
        while True:
            resp = input("Novo rótulo (s/p/n, Enter = manter, q = sair): ").strip().lower()
            if resp in mapa or resp in ("", "q"):
                break
        if resp == "q":
            break
        if resp:
            df.loc[i, "sustenta"] = mapa[resp]
            df.to_csv(ARQ_CITACOES, index=False, encoding="utf-8-sig")
    print("\nProgresso salvo. Para ver os numeros: python src/avaliar.py --resumo")


# --- Modo 3: resumo ----------------------------------------------------------------------
def resumo() -> None:
    if not ARQ_RESULTADOS.exists():
        raise SystemExit("Rode antes: python src/avaliar.py --executar")
    df = pd.read_csv(ARQ_RESULTADOS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out = []

    def p(texto=""):
        out.append(texto)

    p("=== Comportamento ===")
    v = df[df["tipo"] == "valida"]
    resp_v = (v["status"] == "respondida").sum()
    p(f"Perguntas válidas: {resp_v}/{len(v)} respondidas; recusas indevidas: {len(v) - resp_v}")
    for _, r in v[v["status"] != "respondida"].iterrows():
        p(f"   recusada: {r['id']} ({r['status']}, melhor sim {r['melhor_similaridade']}): {r['pergunta']}")
    e = df[df["tipo"] == "fora_escopo"]
    if len(e):
        cont = e["status"].value_counts()
        recusadas = e["status"].isin(["fora_do_escopo", "sem_evidencia"]).sum()
        p(f"Fora do escopo: {len(e)} execuções ({e['id'].nunique()} perguntas); "
          f"recusadas {recusadas}/{len(e)}")
        p(f"   pelo classificador de escopo (fora_do_escopo): {cont.get('fora_do_escopo', 0)}")
        p(f"   pelo limiar ou pelo modelo (sem_evidencia):     {cont.get('sem_evidencia', 0)}")
        p(f"   respondidas indevidamente:                      {cont.get('respondida', 0)}")
        for id_, g in e.groupby("id"):
            p(f"   {id_}: " + ", ".join(f"{k} x{n}" for k, n in g["status"].value_counts().items()))
    nv = df[df["id"].isin(NOVAS)]
    if len(nv):
        nv_v, nv_e = nv[nv["tipo"] == "valida"], nv[nv["tipo"] == "fora_escopo"]
        p("Perguntas novas da rodada 2 (não usadas para ajustar o sistema):")
        p(f"   válidas respondidas: {(nv_v['status'] == 'respondida').sum()}/{len(nv_v)} | "
          f"fora do escopo recusadas: "
          f"{nv_e['status'].isin(['fora_do_escopo', 'sem_evidencia']).sum()}/{len(nv_e)} execuções")
    n3 = df[df["id"].isin(NOVAS_R3)]
    if len(n3):
        n3_v, n3_e = n3[n3["tipo"] == "valida"], n3[n3["tipo"] == "fora_escopo"]
        p("Perguntas novas da rodada 3 (não usadas para ajustar o sistema):")
        p(f"   válidas respondidas: {(n3_v['status'] == 'respondida').sum()}/{len(n3_v)} | "
          f"fora do escopo recusadas: "
          f"{n3_e['status'].isin(['fora_do_escopo', 'sem_evidencia']).sum()}/{len(n3_e)} execuções")
    aj = df[df["id"].isin(AJUSTE_R3)]
    for _, r in aj.iterrows():
        p(f"Pergunta que motivou o ajuste da rodada 3: {r['id']} -> {r['status']}")
    d_ = df[df["tipo"] == "dificil"]
    if len(d_):
        p("Perguntas difíceis (só registradas; avalie o texto em resultados.csv):")
        for id_, g in d_.groupby("id"):
            p(f"   {id_} ({g['pergunta'].iloc[0]}): "
              + ", ".join(f"{k} x{n}" for k, n in g["status"].value_counts().items()))

    ans = df[df["status"] == "respondida"]
    p("\n=== Tempos das respostas (s, com o modelo já carregado) ===")
    tt = pd.to_numeric(ans["t_total"], errors="coerce").dropna()
    if len(tt):
        p(f"Total: mediana {statistics.median(tt):.1f} | média {tt.mean():.1f} | máx. {tt.max():.1f}")
    for col, nome in (("t_reescrita", "reescrita"), ("t_busca", "busca"), ("t_geracao", "geração")):
        x = pd.to_numeric(ans[col], errors="coerce").dropna()
        if len(x):
            p(f"   {nome}: mediana {statistics.median(x):.1f}")

    p("\n=== Guardas (respostas geradas) ===")
    p(f"Respostas: {len(ans)} | evidências por resposta (média): "
      f"{pd.to_numeric(ans['evidencias'], errors='coerce').mean():.1f}")
    p(f"Palavras de frequência detectadas: {(ans['freq_detectada'] != '').sum()} respostas; "
      f"persistiram após a reescrita: {(ans['freq_apos_reescrita'] != '').sum()}")
    p(f"Citações inexistentes: {(ans['citacoes_invalidas'] != '').sum()} respostas | "
      f"respostas sem nenhuma citação: {(ans['citacoes_validas'] == '').sum()}")

    p("\n=== Precisão das citações (avaliação manual) ===")
    if ARQ_CITACOES.exists():
        c = pd.read_csv(ARQ_CITACOES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        rot = c[c["sustenta"] != ""]
        if len(rot):
            n = len(rot)
            s, pa, na = (rot["sustenta"] == "sim").sum(), (rot["sustenta"] == "parcial").sum(), \
                        (rot["sustenta"] == "nao").sum()
            p(f"Pares afirmação-citação rotulados: {n} de {len(c)} (amostra aleatória)")
            p(f"   sustenta: {s} | parcialmente: {pa} | não sustenta: {na}")
            p(f"   precisão estrita (só 'sim'):   {s / n:.0%}")
            p(f"   precisão tolerante (sim+parcial): {(s + pa) / n:.0%}")
        else:
            p("Nenhum par rotulado ainda. Rode: python src/avaliar.py --rotular")
    else:
        p("Arquivo de citações não encontrado.")

    texto = "\n".join(out)
    print(texto)
    ARQ_RESUMO.write_text(texto, encoding="utf-8")
    print(f"\n(gravado em {ARQ_RESUMO})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliacao do RAG (Fase 5)")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--executar", action="store_true", help="roda as perguntas de teste")
    modo.add_argument("--rotular", action="store_true", help="marca se as citacoes sustentam")
    modo.add_argument("--resumo", action="store_true", help="calcula os numeros finais")
    modo.add_argument("--revisar", action="store_true",
                      help="revisa os pares que tem um dado rotulo (veja --rotulo)")
    ap.add_argument("--rotulo", choices=["sim", "parcial", "nao"], default="nao",
                    help="rotulo a revisar com --revisar (padrao: nao)")
    ap.add_argument("--rodada", default="",
                    help="nome da rodada (pasta avaliacao/<nome>/); vazio = avaliacao/")
    ap.add_argument("--repeticoes", type=int, default=1,
                    help="repeticoes das perguntas validas e dificeis (fora do escopo: sempre 3)")
    args = ap.parse_args()
    definir_rodada(args.rodada)
    if args.executar:
        executar(args.repeticoes)
    elif args.rotular:
        rotular()
    elif args.revisar:
        revisar(args.rotulo)
    else:
        resumo()


if __name__ == "__main__":
    main()
