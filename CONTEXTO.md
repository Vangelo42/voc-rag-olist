# CONTEXTO DO PROJETO (leia isto primeiro)

## Instruções para o assistente
- Responda sempre em português do Brasil.
- O usuário usa Windows 10, Python 3.11 (ambiente `.venv` em `C:\Users\Eu\voc-rag`), VS Code ou CMD.
- Hardware: 32 GB de RAM e 8 GB de VRAM. Tudo deve rodar localmente e de graca.
- Trabalhe em passos pequenos: um arquivo ou um problema por vez. Entregue arquivos completos, nao trechos soltos.
- Nunca invente dados, caminhos ou nomes de funcoes. Se faltar informacao, pergunte.
- Se for corrigir um erro, peca a mensagem de erro completa e o arquivo envolvido.

## O projeto
Tech Challenge Fase 4 (POSTECH): sistema de Voice of Customer com RAG sobre as avaliacoes da base Olist (Brazilian E-Commerce, Kaggle).
**Prazo de entrega: 17/11/2026.**

Entregaveis: repositorio publico no GitHub (codigo, README completo com arquitetura, decisoes, exemplos, instrucoes de reproducao) e video de ate 7 minutos.

Requisitos principais:
1. Base de conhecimento justificada (selecao, limpeza, ausentes, documentos, embeddings, indexacao).
2. Pipeline RAG: transformacao da pergunta, recuperacao semantica, selecao, construcao do contexto, geracao.
3. Respostas fundamentadas, mostrando as evidencias (avaliacoes) usadas.
4. Tratar perguntas sem evidencia suficiente ou fora do escopo, sem inventar.
5. Opcional: metadados (nota, periodo, categoria) para filtros.

## Stack escolhida
- LLM de geracao: Ollama com `qwen3:8b` (API local em http://localhost:11434).
- Embeddings: `intfloat/multilingual-e5-base` (sentence-transformers). Prefixos obrigatorios: `passage: ` nos documentos e `query: ` nas perguntas. Vetores normalizados.
- Banco vetorial: ChromaDB local (`data/chroma_db`), colecao `avaliacoes_olist`, distancia de cosseno (similaridade = 1 - distancia).
- Interface (a fazer): Streamlit.
- O Odysseus (workspace local do PewDiePie) e opcional: so serve para testar prompts e modelos. O codigo do projeto NAO depende dele.

## Estrutura de pastas
```
voc-rag\
  .venv\
  data\
    olist_order_reviews_dataset.csv  (e demais CSVs do Kaggle)
    processed\reviews_processadas.csv
    chroma_db\            (gerado pelo indexar.py)
  src\
    config.py             (caminhos, modelo, prefixos)
    preparar_base.py      (Fase 1)
    indexar.py            (Fase 2)
    buscar_teste.py       (Fase 2)
    calibrar_limiar.py    (Fase 3: calibracao do limiar)
    rag.py                (Fase 3: pipeline RAG, linha de comando)
  explorar.py
```

## Decisoes tomadas (com numeros reais)
- Base original: 99.224 avaliacoes; so 40.977 (41%) tem comentario. Titulo vazio em 88%.
- Funil: 40.977 -> 40.950 (textos vazios) -> 40.641 (review_id duplicado) -> **37.994** (minimo de 2 palavras e 3 letras).
- Comentarios curtos (mediana 60 caracteres, maximo 229): **um documento por avaliacao, sem chunking**.
- Documento = titulo + comentario (titulo so quando existe). Sem remover acentos nem stopwords.
- Textos repetidos (3.592) mantidos de proposito: a frequencia importa. Na recuperacao, textos repetidos e avaliacoes quase iguais sao removidos pelo `rag.py` (ver Decisoes da Fase 2 e 3).
- Metadados no indice: `nota`, `faixa_nota` (negativa 1-2, neutra 3, positiva 4-5), `ano_mes`, `data_avaliacao`, `order_id`.
- Vies a registrar no README: comentam 77% dos clientes de nota 1 e so 36% dos de nota 5.
- Periodo dos dados: 2016-10 a 2018-08.

## Decisoes da Fase 2 e 3 (com numeros reais)
- Repositorio publico: github.com/Vangelo42/voc-rag-olist (`.gitignore` exclui `.venv/` e `data/`).
- Indexacao: 37.994 documentos; embeddings e5-base na CPU em 469 s (~8 min); ChromaDB com cosseno em `data/chroma_db`, colecao `avaliacoes_olist`; cache dos vetores em `data/processed/embeddings_e5_base.npy`.
- Calibracao (`calibrar_limiar.py`, 16 perguntas dentro e 14 fora do escopo): o E5 comprime as similaridades entre ~0,78 e 0,94. Top-1 dentro do escopo: min. 0,845, mediana 0,896, max. 0,941. Fora do escopo: min. 0,787, mediana 0,839, max. 0,850.
- Regra escolhida: responder so com **>= 3 textos unicos com similaridade >= 0,85**. Resultado na calibracao: 94% das validas respondidas (15/16) e 100% das fora do escopo recusadas (14/14). Faixa boa estreita (~0,845-0,85) e calibrada com so 30 perguntas: registrar como limitacao.
- Perguntas genericas ("principais problemas") ficam perto da linha (top-1 0,845 sem reescrita). A reescrita da pergunta pelo LLM sobe a melhor similaridade para 0,876.
- Pipeline (`rag.py`): (1) o LLM reescreve a pergunta em 3-5 frases no estilo de avaliacoes e classifica fora do escopo (FORA_DO_ESCOPO); (2) busca de 30 candidatos por consulta (original + frases), com filtro opcional por faixa de nota; (3) limiar 0,85 + minimo de 3 textos unicos, senao recusa sem chamar o modelo de resposta; (4) selecao de ate 10 evidencias por revezamento entre as consultas, ignorando textos repetidos e avaliacoes quase iguais (cosseno >= `NEAR_DUP` = 0,93); (5) prompt que so permite usar as avaliacoes numeradas, exige citacoes [n] (max. 3 por afirmacao), proibe percentuais, comparacao de frequencia e inferencia de causas; (6) checagem de citacoes (validas x inexistentes).
- Defesas em camadas contra alucinacao: classificador de escopo (LLM) -> limiar e minimo de evidencias -> regras do prompt -> checagem de citacoes.
- Testes reais da Fase 3 (qwen3:8b via Ollama), versao 1 do rag.py:
  - "principais problemas" -> respondida; temas e citacoes corretos, com 2 extrapolacoes leves (inferiu "dificuldade de devolucao" e citou uma avaliacao de entrega como atendimento).
  - "o que os clientes elogiam" (faixa positiva) -> as 10 evidencias eram todas variacoes de "entregou antes do prazo": faltou diversidade. Corrigido na v2 (selecao por revezamento + filtro de quase iguais).
  - "problemas de entrega" (faixa negativa) -> respondida, mas usou "problema mais citado" (afirmacao de frequencia sem base). Corrigido na v2 (regra 3 do prompt).
  - "previsao do tempo" -> `fora_do_escopo` ja na etapa de reescrita (a busca nem rodou).
- Limitacao a registrar no README: "principais" significa as melhores avaliacoes por similaridade, nao as mais frequentes; a resposta descreve uma amostra recuperada, nao uma estatistica.

## Status
- [x] Fase 0: Ollama + qwen3:8b respondendo em portugues; Odysseus instalado e conectado ao Ollama.
- [x] Fase 1: `preparar_base.py` executado com sucesso.
- [x] Fase 2: indexacao concluida (37.994 documentos) e busca testada.
- [~] Fase 3: `calibrar_limiar.py` e `rag.py` v1 prontos, testados com o modelo real e commitados. **A v2 (selecao diversificada, prompt reforcado, tempos no --debug) esta escrita e testada so com simulacao; falta rodar com o qwen3:8b e commitar.**
- [ ] Fase 4: filtros por metadados (incluir categoria do produto via orders, order_items e products) e interface Streamlit.
- [ ] Fase 5: ~20 perguntas de teste, incluindo fora do escopo.
- [ ] Fase 6: README e repositorio.
- [ ] Fase 7: video de ate 7 minutos.
- [ ] Trilha Databricks (+10 pontos) em paralelo.

## Proximo passo
1. Substituir `src/rag.py` pela v2 e re-rodar com `--debug`: "O que os clientes elogiam?" (--faixa positiva), "Quais problemas aparecem nas avaliacoes de clientes insatisfeitos com a entrega?" (--faixa negativa) e "Quais os principais problemas relatados pelos clientes?". Conferir `evidencias_selecionadas` (se ficar abaixo de ~5, subir `NEAR_DUP` para 0,95) e se as evidencias agora cobrem varios temas.
2. Testar a camada do limiar sem a reescrita: `--sem-reescrita` com "Qual a previsao do tempo para amanha?" (esperado: `sem_evidencia`). Com a reescrita, testar as perguntas dificeis "Quantos pedidos a empresa teve em 2017?" e "Qual o telefone do suporte da loja?" para ver se o classificador de escopo as barra.
3. Anotar os tempos (`tempos_s` no --debug) para o README.
4. Commitar e seguir para a Fase 4 (filtros por nota/periodo/categoria e interface Streamlit).

## Registro de trocas (preencher a cada troca de assistente)
- Data / quem trabalhou / o que foi feito / o que ficou pendente:
