"""Fase 3/4 - Pipeline RAG (Voice of Customer sobre avaliacoes da Olist).

Fluxo:
  pergunta -> processamento (LLM reescreve, checa escopo e sugere categoria;
  regras inferem filtros de nota, entrega e periodo) -> embeddings
  -> recuperacao no ChromaDB com filtros de metadados
  -> limiar de relevancia (>= N evidencias unicas)
  -> selecao diversificada -> contexto numerado -> geracao (Ollama)
  -> guarda de palavras de frequencia (v3) -> checagem de citacoes

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo e o Ollama aberto):
    python src/rag.py "Quais os principais problemas de entrega?"
    python src/rag.py "O que os clientes elogiam?" --debug
    python src/rag.py "Problemas de entrega em 2017" --debug   # filtros inferidos da pergunta
    python src/rag.py "O que dizem?" --faixa negativa --entrega atrasada --categoria beleza_saude
    python src/rag.py --listar-categorias
    python src/rag.py                      # modo interativo
"""
import argparse
import os
import re
import time
import unicodedata

import chromadb
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLECAO, MODELO_EMBEDDINGS, PREFIXO_PERGUNTA
from filtros import descrever, inferir_filtros, montar_where, validar_categoria

# --- Parametros do pipeline (justificar no README) -------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = "qwen3:8b"
LIMIAR = 0.85          # similaridade minima para uma avaliacao contar como evidencia
MIN_EVIDENCIAS = 3     # textos unicos acima do limiar para tentar responder
TOP_K_BUSCA = 30       # candidatos por consulta (busca larga, depois filtra e seleciona)
MAX_EVIDENCIAS = 10    # evidencias enviadas ao modelo
NEAR_DUP = 0.93        # similaridade entre evidencias a partir da qual sao "quase iguais"
TIMEOUT_S = 300
LOTE_LEITURA = 2000    # tamanho do lote ao ler os metadados do indice (limite do SQLite)

ROTULOS_FILTROS = {"faixa_nota": "faixa de nota", "nota": "nota", "faixa_entrega": "entrega",
                   "categoria": "categoria", "meses": "período da compra"}

MSG_SEM_EVIDENCIA = (
    "Não encontrei evidências suficientes nas avaliações para responder a essa pergunta."
)
MSG_FORA_ESCOPO = (
    "Essa pergunta está fora do escopo desta base, que contém apenas avaliações escritas "
    "por clientes (entrega, produto, atendimento e experiência de compra)."
)

PROMPT_REESCRITA = """Você ajuda a buscar avaliações de clientes de uma loja online brasileira.
Dada a PERGUNTA de um gestor, responda neste formato exato:

Linha 1: ESCOPO: DENTRO - <motivo em até 8 palavras>   ou   ESCOPO: FORA - <motivo em até 8 palavras>
- DENTRO: a pergunta quer saber O QUE os clientes escreveram nas avaliações: compras, entrega, \
produtos, vendedores, atendimento, qualidade, elogios, reclamações, problemas, padrões e \
experiência dos clientes.
- Ano, mês, nota ou categoria de produto na pergunta NÃO a tiram do escopo: o sistema filtra \
as avaliações por eles. Uma pergunta sobre o que os clientes escreveram em um período é DENTRO.
- FORA: assuntos sem relação com avaliações de clientes (esportes, notícias, receitas, \
programação, conhecimentos gerais...) ou perguntas cuja resposta é um NÚMERO ou um dado da \
empresa que não está escrito nas avaliações: quantos, quanto, qual o total, média, \
faturamento, preços, telefones, endereços ou dados cadastrais.
Exemplos:
"Que queixas sobre troca e devolução surgiram em maio de 2017?" -> ESCOPO: DENTRO - relatos de clientes em um período
"Qual foi o valor médio do frete em 2017?" -> ESCOPO: FORA - pede um número
Na dúvida, escolha DENTRO.
Se for FORA, pare na linha 1 e não escreva mais nada.

Se for DENTRO, depois da linha 1 escreva de 3 a 5 frases curtas, no estilo de avaliações \
reais de clientes, que seriam boas respostas para a pergunta (uma frase por linha, sem \
numeração e sem explicações). Na última linha escreva CATEGORIA: <nome>.
Escreva CATEGORIA: nenhuma, a menos que a pergunta cite explicitamente um tipo de produto que \
corresponda a uma das categorias abaixo. Não deduza categoria a partir de temas como entrega, \
atendimento, preço ou qualidade em geral. Se citar, copie o nome exato da lista.
CATEGORIAS: {categorias}

PERGUNTA: {pergunta}"""

SISTEMA_RESPOSTA = """Você é um analista de Voice of Customer. Responda à PERGUNTA usando SOMENTE \
as AVALIAÇÕES numeradas fornecidas.
Regras:
1. Use apenas o que está escrito nas avaliações. Não use conhecimento externo, não invente fatos \
e não infira causas ou detalhes que as avaliações não mencionam.
2. Cite as avaliações que sustentam cada afirmação com o número entre colchetes, por exemplo [2][5]. \
Cite no máximo 3 por afirmação, escolhendo as mais representativas.
3. As avaliações são uma amostra recuperada, não uma estatística. Não informe percentuais ou totais \
e não compare a frequência entre temas: nada de "a maioria", "muitos", "o mais citado" ou \
"principal problema". Escreva "há relatos de ...".
4. Se as avaliações não responderem à pergunta, responda exatamente: \
Não encontrei evidências suficientes nas avaliações para responder a essa pergunta.
5. Responda em português do Brasil, em no máximo 8 linhas: uma frase de síntese e depois \
tópicos curtos, um por tema."""


# --- Funcoes auxiliares ------------------------------------------------------
def normalizar(texto: str) -> str:
    """Minusculas, sem acentos nem pontuacao: usado para achar textos repetidos."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def selecionar_diversas(listas: list[list[dict]], limiar: float, maximo: int,
                        near_dup: float = NEAR_DUP) -> list[dict]:
    """Escolhe evidencias variadas.

    Reveza entre as consultas (a original e as frases geradas na reescrita), pegando a
    melhor avaliacao ainda nao usada de cada uma, para que todos os temas aparecam.
    Ignora textos repetidos e avaliacoes quase iguais (cosseno >= near_dup) as ja escolhidas.
    Cada item de `listas` esta ordenado da maior para a menor similaridade.
    """
    filas = [[c for c in lista if c["sim"] >= limiar] for lista in listas]
    posicoes = [0] * len(filas)
    escolhidas, ids, textos = [], set(), set()
    while len(escolhidas) < maximo:
        avancou = False
        for q, fila in enumerate(filas):
            if len(escolhidas) >= maximo:
                break
            while posicoes[q] < len(fila):
                cand = fila[posicoes[q]]
                posicoes[q] += 1
                chave = normalizar(cand["doc"])
                if cand["id"] in ids or chave in textos:
                    continue
                if any(float(cand["emb"] @ e["emb"]) >= near_dup for e in escolhidas):
                    continue
                escolhidas.append(cand)
                ids.add(cand["id"])
                textos.add(chave)
                avancou = True
                break
        if not avancou:
            break
    return escolhidas


def chamar_ollama(mensagens: list[dict], modelo: str, temperatura: float = 0.1) -> str:
    corpo = {
        "model": modelo,
        "messages": mensagens,
        "stream": False,
        "keep_alive": "30m",  # mantem o modelo na memoria entre perguntas (evita ~20 s de recarga)
        "think": False,  # modelos com raciocinio (ex.: Qwen3): resposta direta e mais rapida
        "options": {"temperature": temperatura, "num_ctx": 4096},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=corpo, timeout=TIMEOUT_S)
        r.raise_for_status()
    except requests.exceptions.ConnectionError as erro:
        raise RuntimeError(
            f"Nao consegui conectar ao Ollama em {OLLAMA_URL}. Abra o Ollama e tente de novo."
        ) from erro
    texto = r.json()["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()


def montar_contexto(evidencias: list[dict]) -> str:
    return "\n".join(
        f"[{i}] (nota {e['meta']['nota']}, compra {e['meta'].get('ano_mes_compra', e['meta']['ano_mes'])}) {e['doc']}"
        for i, e in enumerate(evidencias, start=1)
    )


def conferir_citacoes(resposta: str, total: int) -> tuple[list[int], list[int]]:
    citadas = {int(n) for n in re.findall(r"\[(\d+)\]", resposta)}
    validas = sorted(n for n in citadas if 1 <= n <= total)
    invalidas = sorted(n for n in citadas if not 1 <= n <= total)
    return validas, invalidas


# --- Guarda de palavras de frequencia (v3) ----------------------------------
# A resposta descreve uma amostra recuperada, nao uma estatistica. Estas expressoes
# comparam frequencia sem base e sao detectadas na resposta (texto normalizado).
PADROES_FREQUENCIA = re.compile(
    r"\b(principalmente|principais|principal|maioria|majoritariamente|predominantemente"
    r"|frequentemente|geralmente|comumente|muitos|muitas|varios|varias"
    r"|mais (?:citad[oa]s?|comum|comuns|frequentes?|recorrentes?|mencionad[oa]s?))\b"
)


def detectar_frequencia(resposta: str, pergunta: str = "") -> list[str]:
    """Expressoes de frequencia na resposta. Ignora as que o proprio usuario usou na pergunta
    (ex.: 'principais problemas'), pois ecoar a pergunta nao e afirmar frequencia."""
    pergunta_norm = normalizar(pergunta)
    achadas = []
    for m in PADROES_FREQUENCIA.finditer(normalizar(resposta)):
        termo = m.group(0)
        if termo not in pergunta_norm and termo not in achadas:
            achadas.append(termo)
    return achadas


def pedir_reescrita_sem_frequencia(mensagens: list[dict], resposta: str,
                                   termos: list[str]) -> list[dict]:
    """Mensagens para a segunda tentativa: mostra a resposta anterior e pede a correcao."""
    return mensagens + [
        {"role": "assistant", "content": resposta},
        {"role": "user", "content": (
            "Reescreva a resposta sem estas expressões de frequência: "
            + ", ".join(f'"{t}"' for t in termos)
            + '. Use "há relatos de ..." e não compare a frequência entre temas. '
            "Mantenha o mesmo formato: uma frase de síntese e depois tópicos curtos, um por tema, "
            "cada tópico com as suas próprias citações [n] no fim da frase. "
            "Mantenha todas as outras regras. Responda só com a nova versão."
        )},
    ]


def saida_indica_fora_do_escopo(saida: str) -> bool:
    """Reconhece 'ESCOPO: FORA - motivo' e variacoes ('FORA', 'ESCOPO:' + 'FORA' na linha
    seguinte, markdown, FORA_DO_ESCOPO). So olha o INICIO da saida: o motivo pode conter a
    palavra 'fora' e uma frase comum que comeca com 'Fora do prazo...' NAO conta."""
    linhas = [normalizar(l).replace("_", " ") for l in saida.splitlines() if l.strip()]
    if not linhas:
        return False
    primeira = linhas[0]
    if re.match(r"escopo fora\b", primeira) or re.fullmatch(r"fora( do escopo)?", primeira):
        return True
    return primeira == "escopo" and len(linhas) > 1 and linhas[1].startswith("fora")


def combinar_filtros(inferidos: dict, manuais: dict | None) -> dict:
    """Filtros inferidos da pergunta + filtros escolhidos pelo usuario (manuais mandam).
    Um filtro manual de nota substitui os de nota inferidos, para nao se contradizerem."""
    final = dict(inferidos)
    manuais = {k: v for k, v in (manuais or {}).items() if v}
    if "faixa_nota" in manuais or "nota" in manuais:
        final.pop("faixa_nota", None)
        final.pop("nota", None)
    final.update(manuais)
    return final


# --- Pipeline ------------------------------------------------------------------
class RAG:
    def __init__(self, modelo_llm=MODELO_LLM, limiar=LIMIAR, min_evidencias=MIN_EVIDENCIAS):
        self.modelo_llm = modelo_llm
        self.limiar = limiar
        self.min_evidencias = min_evidencias
        self.encoder = SentenceTransformer(MODELO_EMBEDDINGS)
        self.colecao = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLECAO)
        # Categorias que existem no indice (o LLM so pode escolher entre elas)
        # Leitura em lotes: pedir tudo de uma vez estoura o limite de variaveis do SQLite
        categorias = set()
        total = self.colecao.count()
        for inicio in range(0, total, LOTE_LEITURA):
            lote = self.colecao.get(include=["metadatas"], limit=LOTE_LEITURA, offset=inicio)
            categorias.update(m["categoria"] for m in lote["metadatas"] if m.get("categoria"))
        self.categorias = sorted(categorias - {"sem_categoria"})

    # Etapa 1: processamento da pergunta
    def processar_pergunta(self, pergunta: str, reescrever: bool = True):
        """Retorna (consultas, fora_do_escopo, categoria).
        Consultas = pergunta original + frases geradas. Categoria = sugestao do LLM, ja validada."""
        consultas = [pergunta]
        if not reescrever:
            return consultas, False, None
        saida = chamar_ollama(
            [{"role": "user", "content": PROMPT_REESCRITA.format(
                pergunta=pergunta, categorias=", ".join(self.categorias))}],
            self.modelo_llm,
        )
        self.ultima_saida_reescrita = saida  # aparece no --debug (saida_reescrita)
        if saida_indica_fora_do_escopo(saida):
            return consultas, True, None
        linhas = [l for l in saida.splitlines() if l.strip()]
        categoria = None
        for linha in linhas:
            if re.match(r"^\W*escopo\s*:", linha.strip(), flags=re.I):
                continue
            if normalizar(linha) in ("dentro", "fora"):  # sobra do formato de escopo
                continue
            m = re.match(r"^\W*(categorias?)\s*:\s*(.*)$", linha.strip(), flags=re.I)
            if m:
                if m.group(1).lower() == "categoria":
                    valor = m.group(2).strip(" *_`'\".")
                    categoria = validar_categoria(valor, self.categorias)  # None se "nenhuma"/invalida
                continue
            frase = re.sub(r"^[\-\*\d\.\)\s]+", "", linha).strip()
            if frase:
                consultas.append(frase)
        return consultas[:6], False, categoria

    # Etapa 2: recuperacao semantica (uma lista ordenada por consulta)
    def buscar(self, consultas: list[str], onde: dict | None = None) -> list[list[dict]]:
        """`onde` = filtro de metadados no formato do ChromaDB (ver filtros.montar_where)."""
        listas = []
        for texto in consultas:
            vetor = self.encoder.encode(PREFIXO_PERGUNTA + texto, normalize_embeddings=True)
            res = self.colecao.query(
                query_embeddings=[vetor.tolist()],
                n_results=TOP_K_BUSCA,
                where=onde,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            listas.append([
                {"id": id_, "doc": doc, "meta": meta, "sim": 1 - dist,
                 "emb": np.asarray(emb, dtype="float32")}
                for id_, doc, meta, dist, emb in zip(
                    res["ids"][0], res["documents"][0], res["metadatas"][0],
                    res["distances"][0], res["embeddings"][0])
            ])
        return listas

    def perguntar(self, pergunta: str, filtros_manuais: dict | None = None,
                  reescrever: bool = True, filtros_automaticos: bool = True) -> dict:
        """filtros_manuais: chaves faixa_nota, nota, faixa_entrega, meses (lista 'AAAA-MM'),
        categoria. Mandam sobre os inferidos. filtros_automaticos=False desliga a inferencia
        (regras de nota/entrega/periodo e a categoria sugerida pelo LLM)."""
        resultado = {
            "pergunta": pergunta, "status": None, "resposta": None, "evidencias": [],
            "citacoes_validas": [], "citacoes_invalidas": [], "avisos": [], "detalhes": {},
            "filtros": {}, "filtros_descricao": "sem filtros",
        }
        tempos = resultado["detalhes"]["tempos_s"] = {}
        inicio = time.perf_counter()

        consultas, fora, categoria_llm = self.processar_pergunta(pergunta, reescrever)
        tempos["reescrita"] = round(time.perf_counter() - inicio, 1)
        resultado["detalhes"]["consultas"] = consultas
        if reescrever:
            resultado["detalhes"]["saida_reescrita"] = getattr(self, "ultima_saida_reescrita", None)
        if fora:
            tempos["total"] = tempos["reescrita"]
            resultado.update(status="fora_do_escopo", resposta=MSG_FORA_ESCOPO)
            return resultado

        # Filtros: regras (nota, entrega, periodo) + categoria do LLM + escolhas manuais
        inferidos, motivos = inferir_filtros(pergunta) if filtros_automaticos else ({}, [])
        if filtros_automaticos and categoria_llm:
            inferidos["categoria"] = categoria_llm
            motivos.append(f"categoria {categoria_llm} (sugerida pelo LLM, validada na lista)")
        manuais = {k: v for k, v in (filtros_manuais or {}).items() if v}
        motivos += [f"filtro de {ROTULOS_FILTROS.get(k, k)} escolhido manualmente" for k in manuais]
        filtros = combinar_filtros(inferidos, manuais)
        onde = montar_where(filtros)
        resultado["filtros"] = filtros
        resultado["filtros_descricao"] = descrever(filtros)
        resultado["detalhes"]["filtros_motivos"] = motivos
        resultado["detalhes"]["where"] = onde

        t = time.perf_counter()
        listas = self.buscar(consultas, onde)
        tempos["busca"] = round(time.perf_counter() - t, 1)

        melhor_por_id: dict[str, dict] = {}
        for lista in listas:
            for c in lista:
                if c["id"] not in melhor_por_id or c["sim"] > melhor_por_id[c["id"]]["sim"]:
                    melhor_por_id[c["id"]] = c
        acima = {i: c for i, c in melhor_por_id.items() if c["sim"] >= self.limiar}
        textos_unicos = {normalizar(c["doc"]) for c in acima.values()}
        melhor = max((lista[0]["sim"] for lista in listas if lista), default=None)
        resultado["detalhes"].update(
            candidatos=len(melhor_por_id),
            acima_do_limiar=len(acima),
            textos_unicos_acima_do_limiar=len(textos_unicos),
            melhor_similaridade=round(melhor, 3) if melhor is not None else None,
        )

        if len(textos_unicos) < self.min_evidencias:
            tempos["total"] = round(time.perf_counter() - inicio, 1)
            resultado.update(status="sem_evidencia", resposta=MSG_SEM_EVIDENCIA)
            if filtros:
                resultado["avisos"].append(
                    f"Havia filtros ativos ({resultado['filtros_descricao']}). "
                    "Tente remover algum deles ou ampliar o período.")
            return resultado

        evidencias = selecionar_diversas(listas, self.limiar, MAX_EVIDENCIAS)
        resultado["detalhes"]["evidencias_selecionadas"] = len(evidencias)

        mensagens = [
            {"role": "system", "content": SISTEMA_RESPOSTA},
            {"role": "user",
             "content": f"PERGUNTA: {pergunta}\n\nAVALIAÇÕES:\n{montar_contexto(evidencias)}"},
        ]
        t = time.perf_counter()
        resposta = chamar_ollama(mensagens, self.modelo_llm)
        tempos["geracao"] = round(time.perf_counter() - t, 1)

        # Guarda de frequencia: uma nova versao, no maximo; se persistir, aviso
        termos = detectar_frequencia(resposta, pergunta)
        resultado["detalhes"]["frequencia_detectada"] = termos
        if termos:
            t = time.perf_counter()
            resposta = chamar_ollama(
                pedir_reescrita_sem_frequencia(mensagens, resposta, termos), self.modelo_llm)
            tempos["reescrita_frequencia"] = round(time.perf_counter() - t, 1)
            restantes = detectar_frequencia(resposta, pergunta)
            resultado["detalhes"]["frequencia_apos_reescrita"] = restantes
            if restantes:
                resultado["avisos"].append(
                    "A resposta ainda contém expressões de frequência "
                    f"({', '.join(restantes)}). As avaliações são uma amostra recuperada, "
                    "não uma estatística.")
        tempos["total"] = round(time.perf_counter() - inicio, 1)

        if normalizar(resposta).startswith(normalizar(MSG_SEM_EVIDENCIA)[:40]):
            resultado.update(status="sem_evidencia", resposta=MSG_SEM_EVIDENCIA)
            return resultado

        validas, invalidas = conferir_citacoes(resposta, len(evidencias))
        if not validas:
            resultado["avisos"].append("A resposta não cita nenhuma evidência; trate com cautela.")
        if invalidas:
            resultado["avisos"].append(f"A resposta cita números inexistentes: {invalidas}.")
        resultado.update(
            status="respondida", resposta=resposta, evidencias=evidencias,
            citacoes_validas=validas, citacoes_invalidas=invalidas,
        )
        return resultado


# --- Linha de comando ------------------------------------------------------------
def imprimir(res: dict, debug: bool) -> None:
    print("\n" + "=" * 72)
    print(f"Pergunta: {res['pergunta']}")
    print(f"Status:   {res['status']}")
    print(f"Filtros:  {res.get('filtros_descricao', 'sem filtros')}")
    print("-" * 72)
    print(res["resposta"])
    for aviso in res["avisos"]:
        print(f"\n[aviso] {aviso}")
    if res["evidencias"]:
        print("\nEvidências (* = citada na resposta):")
        for i, e in enumerate(res["evidencias"], start=1):
            marca = "*" if i in res["citacoes_validas"] else " "
            print(f" {marca}[{i}] sim={e['sim']:.3f} | nota {e['meta']['nota']} "
                  f"| compra {e['meta'].get('ano_mes_compra', '?')} | aval. {e['meta']['ano_mes']} "
                  f"| {e['doc']}")
    if debug:
        print("\n[debug]")
        for k, v in res["detalhes"].items():
            print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG sobre avaliacoes de clientes (Olist)")
    ap.add_argument("pergunta", nargs="*", help="pergunta (vazio = modo interativo)")
    ap.add_argument("--faixa", choices=["negativa", "neutra", "positiva"],
                    help="filtro manual de faixa de nota (1-2 negativa, 3 neutra, 4-5 positiva)")
    ap.add_argument("--entrega", choices=["atrasada", "nao_entregue", "no_prazo"],
                    help="filtro manual de entrega (entrega ao cliente x data estimada)")
    ap.add_argument("--categoria", help="filtro manual de categoria (veja --listar-categorias)")
    ap.add_argument("--meses", help="filtro manual de meses da compra, ex.: 2018-03,2018-04")
    ap.add_argument("--sem-filtros", action="store_true",
                    help="desliga os filtros inferidos da pergunta (so valem os manuais)")
    ap.add_argument("--listar-categorias", action="store_true",
                    help="mostra as categorias disponiveis e sai")
    ap.add_argument("--sem-reescrita", action="store_true",
                    help="nao usa o LLM para reescrever a pergunta nem checar o escopo")
    ap.add_argument("--modelo", default=MODELO_LLM, help="modelo do Ollama")
    ap.add_argument("--debug", action="store_true", help="mostra etapas, contagens e tempos")
    args = ap.parse_args()

    rag = RAG(modelo_llm=args.modelo)
    if args.listar_categorias:
        print("\n".join(rag.categorias))
        return

    categoria = None
    if args.categoria:
        categoria = validar_categoria(args.categoria, rag.categorias)
        if categoria is None:
            raise SystemExit(f"Categoria '{args.categoria}' nao existe. "
                             "Use --listar-categorias para ver as opcoes.")
    meses = None
    if args.meses:
        meses = [m.strip() for m in args.meses.split(",") if m.strip()]
        if not all(re.fullmatch(r"\d{4}-\d{2}", m) for m in meses):
            raise SystemExit("--meses deve ter o formato AAAA-MM, separado por virgulas.")
    manuais = {"faixa_nota": args.faixa, "faixa_entrega": args.entrega,
               "categoria": categoria, "meses": meses}
    reescrever = not args.sem_reescrita
    automaticos = not args.sem_filtros

    if args.pergunta:
        imprimir(rag.perguntar(" ".join(args.pergunta), manuais, reescrever, automaticos),
                 args.debug)
        return
    while True:
        p = input("\nPergunta (vazio para sair): ").strip()
        if not p:
            break
        imprimir(rag.perguntar(p, manuais, reescrever, automaticos), args.debug)


if __name__ == "__main__":
    main()
