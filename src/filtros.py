"""Fase 4 (passo 5) - Filtros inferidos da pergunta (regras, sem LLM).

Le a pergunta e devolve filtros sobre os metadados do indice:
  faixa_nota   : negativa | neutra | positiva  (ex.: "problemas" -> negativa)
  nota         : 1 a 5                         (ex.: "nota 1")
  faixa_entrega: atrasada | nao_entregue | no_prazo
  meses        : lista de "AAAA-MM" da COMPRA   (ex.: "em 2017", "marco de 2018")
  categoria    : escolhida pelo LLM no rag.py e validada aqui (validar_categoria)

Regras sao deterministicas e previsiveis: nao dependem do Ollama e nao "inventam" filtros.
Na duvida (palavras positivas e negativas juntas, por exemplo) nenhum filtro e aplicado.

Uso (sem Ollama e sem carregar modelos):
    python src/filtros.py                       # roda os autotestes
    python src/filtros.py "Problemas de entrega em marco de 2018"
"""
import re
import sys
import unicodedata

# Anos com dados na base (compras de 2016-09 a 2018-08). Usado quando a pergunta cita
# so o mes ("em dezembro"): vale para todos os anos.
ANOS_DA_BASE = (2016, 2017, 2018)

MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}
_NOMES_MESES = "|".join(MESES)

# Palavras (inicio de palavra, sem acento) que indicam o tom da pergunta.
_NEGATIVAS = re.compile(
    r"\b(problem|reclam|insatisf|queix|critic|defeit|ruim|ruins|negativ|decepc|pior)"
)
_POSITIVAS = re.compile(
    r"\b(elogi|gost|satisf|positiv|adora|recomend|valoriz)"
)
_ATRASO = re.compile(r"\batras")
_NAO_ENTREGUE = re.compile(
    r"\b(nao entregue|nao foram entregues|nao chegou|nao chegaram|nao recebi|"
    r"nao recebeu|nao receberam|extravi)"
)
_NO_PRAZO = re.compile(r"\b(no prazo|antes do prazo|adiantad)")


def _sem_acentos(texto: str) -> str:
    """Minusculas e sem acentos (a pontuacao e mantida: '03/2018' precisa da barra)."""
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _meses_do_ano(ano: int) -> list[str]:
    return [f"{ano}-{m:02d}" for m in range(1, 13)]


# --- Nota ---------------------------------------------------------------------------
def _inferir_nota(t: str, motivos: list[str]) -> dict:
    # "nota 1", "notas 1 e 2", "nota 4 ou 5"
    m = re.search(r"\bnotas?\s+([1-5])(?:\s*(?:e|ou|,)\s*([1-5]))?\b", t)
    if m:
        valores = {int(v) for v in m.groups() if v}
        if len(valores) == 1:
            nota = valores.pop()
            motivos.append(f"nota = {nota} (trecho '{m.group(0)}')")
            return {"nota": nota}
        if valores <= {1, 2}:
            motivos.append(f"faixa de nota negativa (trecho '{m.group(0)}')")
            return {"faixa_nota": "negativa"}
        if valores <= {4, 5}:
            motivos.append(f"faixa de nota positiva (trecho '{m.group(0)}')")
            return {"faixa_nota": "positiva"}
        return {}
    m = re.search(r"\bnotas?\s+(baixa|baixas|ruim|ruins|negativa|negativas)\b", t)
    if m:
        motivos.append(f"faixa de nota negativa (trecho '{m.group(0)}')")
        return {"faixa_nota": "negativa"}
    m = re.search(r"\bnotas?\s+(alta|altas|boa|boas|maxima|positiva|positivas)\b", t)
    if m:
        motivos.append(f"faixa de nota positiva (trecho '{m.group(0)}')")
        return {"faixa_nota": "positiva"}

    neg, pos = _NEGATIVAS.search(t), _POSITIVAS.search(t)
    if neg and not pos:
        motivos.append(f"faixa de nota negativa (palavra '{neg.group(0)}...')")
        return {"faixa_nota": "negativa"}
    if pos and not neg:
        motivos.append(f"faixa de nota positiva (palavra '{pos.group(0)}...')")
        return {"faixa_nota": "positiva"}
    return {}


# --- Entrega --------------------------------------------------------------------------
def _inferir_entrega(t: str, motivos: list[str]) -> dict:
    achados = {}
    if _ATRASO.search(t):
        achados["atrasada"] = "atraso"
    if _NAO_ENTREGUE.search(t):
        achados["nao_entregue"] = "pedido nao entregue"
    if _NO_PRAZO.search(t):
        achados["no_prazo"] = "no prazo"
    if len(achados) == 1:  # mais de um sinal = pergunta ambigua: nao filtra
        faixa, rotulo = next(iter(achados.items()))
        motivos.append(f"faixa de entrega {faixa} (sinal: {rotulo})")
        return {"faixa_entrega": faixa}
    return {}


# --- Periodo (da compra) --------------------------------------------------------------
def _inferir_periodo(t: str, motivos: list[str]) -> dict:
    anos_no_texto = sorted({int(a) for a in re.findall(r"\b(20\d{2})\b", t)})
    meses: set[str] = set()

    def anos_para(ano_proprio: str | None) -> tuple:
        if ano_proprio:
            return (int(ano_proprio),)
        if len(anos_no_texto) == 1:
            return (anos_no_texto[0],)
        return ANOS_DA_BASE

    # 1) formatos numericos: 03/2018, 3-2018, 2018-03
    for m in re.finditer(r"\b(0?[1-9]|1[0-2])\s*[/-]\s*(20\d{2})\b", t):
        meses.add(f"{int(m.group(2))}-{int(m.group(1)):02d}")
    for m in re.finditer(r"\b(20\d{2})\s*-\s*(0[1-9]|1[0-2])\b", t):
        meses.add(f"{int(m.group(1))}-{int(m.group(2)):02d}")

    # 2) intervalo: "de marco a junho de 2018"
    intervalo = re.search(
        rf"\bde\s+({_NOMES_MESES})(?:\s+de\s+(20\d{{2}}))?\s+(?:a|ate)\s+({_NOMES_MESES})"
        rf"(?:\s+de\s+(20\d{{2}}))?\b", t)
    texto_restante = t
    if intervalo:
        m1, a1, m2, a2 = intervalo.groups()
        if MESES[m1] <= MESES[m2]:
            for ano in anos_para(a2 or a1):
                for mes in range(MESES[m1], MESES[m2] + 1):
                    meses.add(f"{ano}-{mes:02d}")
            texto_restante = t[:intervalo.start()] + " " + t[intervalo.end():]

    # 3) meses por extenso, com ano opcional: "marco de 2018", "dezembro"
    for m in re.finditer(
            rf"\b({_NOMES_MESES})\b(?:\s*(?:de|/|-)?\s*(20\d{{2}})\b)?", texto_restante):
        for ano in anos_para(m.group(2)):
            meses.add(f"{ano}-{MESES[m.group(1)]:02d}")

    # 4) so ano(s), sem nenhum mes: periodo = ano inteiro
    if not meses and anos_no_texto:
        for ano in anos_no_texto:
            meses.update(_meses_do_ano(ano))

    if not meses:
        return {}
    lista = sorted(meses)
    resumo = f"{lista[0]} a {lista[-1]}" if len(lista) > 2 else ", ".join(lista)
    motivos.append(f"periodo da compra: {len(lista)} mes(es) ({resumo})")
    return {"meses": lista}


# --- API publica -----------------------------------------------------------------------
def inferir_filtros(pergunta: str) -> tuple[dict, list[str]]:
    """Retorna (filtros, motivos). 'motivos' diz por que cada filtro foi aplicado
    (util para o --debug e para mostrar na interface)."""
    t = _sem_acentos(pergunta)
    motivos: list[str] = []
    filtros: dict = {}
    filtros.update(_inferir_nota(t, motivos))
    filtros.update(_inferir_entrega(t, motivos))
    filtros.update(_inferir_periodo(t, motivos))
    return filtros, motivos


def validar_categoria(nome: str | None, categorias_validas: list[str]) -> str | None:
    """Confere o nome sugerido pelo LLM contra as categorias que existem no indice.
    Devolve o nome exato da categoria ou None (nunca um nome inventado)."""
    if not nome:
        return None
    alvo = _sem_acentos(nome).strip().replace(" ", "_")
    for cat in categorias_validas:
        if _sem_acentos(cat) == alvo:
            return cat
    return None


def montar_where(filtros: dict) -> dict | None:
    """Converte os filtros no formato `where` do ChromaDB."""
    condicoes = []
    for chave in ("faixa_nota", "nota", "faixa_entrega", "categoria"):
        if filtros.get(chave) is not None:
            condicoes.append({chave: filtros[chave]})
    if filtros.get("meses"):
        condicoes.append({"ano_mes_compra": {"$in": list(filtros["meses"])}})
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


def descrever(filtros: dict) -> str:
    """Texto curto para mostrar ao usuario."""
    partes = []
    if filtros.get("faixa_nota"):
        partes.append(f"nota {filtros['faixa_nota']}")
    if filtros.get("nota") is not None:
        partes.append(f"nota {filtros['nota']}")
    if filtros.get("faixa_entrega"):
        partes.append(f"entrega {filtros['faixa_entrega']}")
    if filtros.get("categoria"):
        partes.append(f"categoria {filtros['categoria']}")
    if filtros.get("meses"):
        m = filtros["meses"]
        partes.append(f"compra em {m[0]}" if len(m) == 1 else f"compra de {m[0]} a {m[-1]}")
    return "; ".join(partes) if partes else "sem filtros"


# --- Autotestes -------------------------------------------------------------------------
def _autotestes() -> None:
    meses_2017 = _meses_do_ano(2017)
    casos = [
        ("Quais os principais problemas relatados pelos clientes?", {"faixa_nota": "negativa"}),
        ("O que os clientes elogiam?", {"faixa_nota": "positiva"}),
        ("Quais reclamações sobre atraso na entrega em 2017?",
         {"faixa_nota": "negativa", "faixa_entrega": "atrasada", "meses": meses_2017}),
        ("Problemas em março de 2018", {"faixa_nota": "negativa", "meses": ["2018-03"]}),
        ("O que elogiam em dezembro?",
         {"faixa_nota": "positiva", "meses": ["2016-12", "2017-12", "2018-12"]}),
        ("Quais elogios e reclamações aparecem?", {}),
        ("Quais problemas aparecem de março a junho de 2018?",
         {"faixa_nota": "negativa", "meses": ["2018-03", "2018-04", "2018-05", "2018-06"]}),
        ("Avaliações com nota 1", {"nota": 1}),
        ("O que dizem nas notas 1 e 2?", {"faixa_nota": "negativa"}),
        ("O que dizem nas notas 4 ou 5?", {"faixa_nota": "positiva"}),
        ("Problemas relatados nas avaliações com nota baixa", {"faixa_nota": "negativa"}),
        ("O que dizem sobre a entrega?", {}),
        ("O que podemos melhorar?", {}),
        ("Clientes insatisfeitos", {"faixa_nota": "negativa"}),
        ("Pedidos que não chegaram", {"faixa_entrega": "nao_entregue"}),
        ("O que dizem sobre entregas antes do prazo?", {"faixa_entrega": "no_prazo"}),
        ("Reclamações de pedidos atrasados que não chegaram", {"faixa_nota": "negativa"}),
        ("Problemas em 03/2018", {"faixa_nota": "negativa", "meses": ["2018-03"]}),
        ("Problemas em 2018-03", {"faixa_nota": "negativa", "meses": ["2018-03"]}),
        ("Elogios em janeiro de 2017 e fevereiro de 2017",
         {"faixa_nota": "positiva", "meses": ["2017-01", "2017-02"]}),
        ("Qual a previsão do tempo em 2019?", {"meses": _meses_do_ano(2019)}),
    ]
    erros = 0
    for pergunta, esperado in casos:
        obtido, _ = inferir_filtros(pergunta)
        ok = obtido == esperado
        erros += not ok
        print(f"[{'OK' if ok else 'ERRO'}] {pergunta}")
        if not ok:
            print(f"       esperado: {esperado}\n       obtido:   {obtido}")
    assert validar_categoria("Beleza Saude", ["beleza_saude", "bebes"]) == "beleza_saude"
    assert validar_categoria("brinquedos_inexistente", ["beleza_saude", "bebes"]) is None
    assert validar_categoria(None, ["bebes"]) is None
    assert montar_where({}) is None
    assert montar_where({"faixa_nota": "negativa"}) == {"faixa_nota": "negativa"}
    assert montar_where({"faixa_nota": "negativa", "meses": ["2018-03"]}) == {
        "$and": [{"faixa_nota": "negativa"}, {"ano_mes_compra": {"$in": ["2018-03"]}}]}
    print(f"\n{len(casos) - erros}/{len(casos)} casos corretos.")
    if erros:
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        pergunta = " ".join(sys.argv[1:])
        filtros, motivos = inferir_filtros(pergunta)
        print("Filtros:", descrever(filtros))
        print("where:  ", montar_where(filtros))
        for m in motivos:
            print(" -", m)
    else:
        _autotestes()


if __name__ == "__main__":
    main()
