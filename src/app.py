"""Fase 4 - Interface Streamlit do Voice of Customer (RAG sobre avaliacoes da Olist).

Mostra a pergunta, os filtros (inferidos da pergunta e editaveis), a resposta, as evidencias
usadas (com a marca das que foram citadas) e o aviso de evidencia insuficiente.

Uso (a partir da pasta voc-rag, com o ambiente virtual ativo e o Ollama aberto):
    streamlit run src/app.py
"""
import re

import streamlit as st

from rag import RAG

st.set_page_config(page_title="Voice of Customer - Olist", layout="wide")

AUTO = "Automático"
NOMES_MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
               "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
# (rotulo do botao, pergunta completa). O rotulo e curto para caber; a pergunta aparece ao
# passar o mouse sobre o botao.
EXEMPLOS = [
    ("Elogios", "O que os clientes elogiam?"),
    ("Atraso na entrega em 2017", "Quais reclamações sobre atraso na entrega apareceram em 2017?"),
    ("Elogios: cama, mesa e banho", "O que os clientes elogiam em cama, mesa e banho?"),
    ("Fora do escopo", "Quantos pedidos a empresa teve em 2017?"),
]


@st.cache_resource(show_spinner="Carregando o modelo de embeddings e o índice (só na primeira vez)...")
def carregar_rag() -> RAG:
    return RAG()


def escapar_cifrao(texto: str) -> str:
    """O Streamlit trata $...$ como formula; avaliacoes e respostas podem ter 'R$'."""
    return texto.replace("$", "\\$")


def escapar_md(texto: str) -> str:
    """Escapa caracteres de markdown em textos de avaliacoes (mostrados como citacao)."""
    return re.sub(r"([\\`*_{}\[\]<>~$|])", r"\\\1", texto)


def usar_exemplo(texto: str) -> None:
    st.session_state["pergunta"] = texto
    st.session_state["executar"] = True


def barra_lateral(rag: RAG) -> tuple[dict, bool]:
    """Desenha os filtros e devolve (filtros_manuais, filtros_automaticos)."""
    st.sidebar.header("Filtros")
    automaticos = st.sidebar.checkbox(
        "Inferir filtros da pergunta", value=True,
        help="Ex.: 'problemas' → notas baixas; 'atraso' → pedidos entregues após a data estimada; "
             "'em 2017' → compras de 2017; categoria citada → essa categoria.")
    st.sidebar.caption("Uma escolha abaixo vale mais do que o filtro inferido. "
                       "'Automático' deixa a pergunta decidir.")

    faixa = st.sidebar.selectbox("Faixa de nota", [AUTO, "negativa", "neutra", "positiva"],
                                 help="negativa = notas 1-2; neutra = 3; positiva = 4-5")
    entrega = st.sidebar.selectbox("Entrega", [AUTO, "atrasada", "nao_entregue", "no_prazo"],
                                   help="Data de entrega ao cliente x data estimada.")
    categoria = st.sidebar.selectbox("Categoria do produto", [AUTO] + rag.categorias)
    ano = st.sidebar.selectbox("Ano da compra", [AUTO, 2016, 2017, 2018])
    mes = st.sidebar.selectbox("Mês da compra", ["Todos"] + NOMES_MESES, disabled=(ano == AUTO))

    meses = None
    if ano != AUTO:
        if mes == "Todos":
            meses = [f"{ano}-{m:02d}" for m in range(1, 13)]
        else:
            meses = [f"{ano}-{NOMES_MESES.index(mes) + 1:02d}"]

    st.sidebar.divider()
    st.sidebar.caption(
        f"Modelo de resposta: {rag.modelo_llm}  \n"
        f"Limiar de similaridade: {rag.limiar:.2f}  \n"
        f"Mínimo de textos distintos: {rag.min_evidencias}")

    manuais = {
        "faixa_nota": None if faixa == AUTO else faixa,
        "faixa_entrega": None if entrega == AUTO else entrega,
        "categoria": None if categoria == AUTO else categoria,
        "meses": meses,
    }
    return manuais, automaticos


def mostrar_evidencias(res: dict) -> None:
    st.subheader(f"Evidências recuperadas ({len(res['evidencias'])})")
    st.caption("As avaliações enviadas ao modelo. As marcadas como citadas sustentam a resposta; "
               "as demais foram recuperadas, mas não usadas.")
    for i, e in enumerate(res["evidencias"], start=1):
        m = e["meta"]
        marca = "✅ citada" if i in res["citacoes_validas"] else "não citada"
        st.markdown(
            f"**\\[{i}\\]** {marca} · nota {m['nota']} · compra {m.get('ano_mes_compra', '?')} · "
            f"avaliada em {m['ano_mes']} · {m.get('categoria', '?')} · "
            f"entrega {str(m.get('faixa_entrega', '?')).replace('_', ' ')} · similaridade {e['sim']:.3f}\n\n"
            f"> {escapar_md(e['doc'])}")


def mostrar_resultado(res: dict, rag: RAG) -> None:
    st.divider()
    st.markdown(f"**Pergunta:** {escapar_cifrao(res['pergunta'])}")
    st.markdown(f"**Filtros aplicados:** {res['filtros_descricao']}")
    motivos = res["detalhes"].get("filtros_motivos", [])
    if motivos:
        with st.expander("Por que esses filtros?"):
            for motivo in motivos:
                st.write(f"- {motivo}")

    if res["status"] == "respondida":
        st.subheader("Resposta")
        st.markdown(escapar_cifrao(res["resposta"]))
        st.caption("A resposta descreve uma amostra de avaliações recuperadas, não uma estatística: "
                   "'principais' significa as mais parecidas com a pergunta, não as mais frequentes.")
    elif res["status"] == "sem_evidencia":
        st.warning(res["resposta"])
        d = res["detalhes"]
        if "melhor_similaridade" in d:
            st.caption(
                f"Melhor similaridade encontrada: {d['melhor_similaridade']}. "
                f"Textos distintos acima do limiar de {rag.limiar:.2f}: "
                f"{d.get('textos_unicos_acima_do_limiar', 0)} (mínimo: {rag.min_evidencias}).")
    else:
        st.info(res["resposta"])

    for aviso in res["avisos"]:
        st.warning(aviso)
    if res["evidencias"]:
        mostrar_evidencias(res)

    with st.expander("Detalhes técnicos (etapas do pipeline)"):
        st.write("**Consultas usadas na busca** (pergunta original + frases geradas pelo modelo):")
        for consulta in res["detalhes"].get("consultas", []):
            st.write(f"- {escapar_md(consulta)}")
        st.json(res["detalhes"], expanded=False)


def main() -> None:
    rag = carregar_rag()
    manuais, automaticos = barra_lateral(rag)

    st.title("Voice of Customer — avaliações da Olist")
    st.write("Pergunte em linguagem natural sobre as avaliações dos clientes. "
             "A resposta só usa avaliações recuperadas e mostra quais foram usadas.")

    st.caption("Exemplos:")
    colunas = st.columns(len(EXEMPLOS))
    for i, (coluna, (rotulo, pergunta_exemplo)) in enumerate(zip(colunas, EXEMPLOS)):
        coluna.button(rotulo, key=f"exemplo_{i}", on_click=usar_exemplo,
                      args=(pergunta_exemplo,), help=pergunta_exemplo)

    with st.form("form_pergunta"):
        st.text_input("Pergunta", key="pergunta",
                      placeholder="Ex.: Quais problemas de entrega apareceram em 2018?")
        enviado = st.form_submit_button("Perguntar")

    if enviado or st.session_state.pop("executar", False):
        pergunta = st.session_state.get("pergunta", "").strip()
        if pergunta:
            try:
                with st.spinner("Buscando avaliações e gerando a resposta..."):
                    st.session_state["resultado"] = rag.perguntar(
                        pergunta, manuais, reescrever=True, filtros_automaticos=automaticos)
            except Exception as erro:  # ex.: Ollama fechado
                st.session_state.pop("resultado", None)
                st.error(f"Não consegui processar a pergunta: {erro}")

    if st.session_state.get("resultado"):
        mostrar_resultado(st.session_state["resultado"], rag)
        st.caption("Alterou os filtros? Clique em Perguntar de novo para aplicar.")


main()
