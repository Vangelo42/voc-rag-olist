# CONTEXTO DO PROJETO (leia isto primeiro)

## Instruções para o assistente
- Responda sempre em português do Brasil.
- O usuário usa Windows 10, Python 3.11 (ambiente `.venv` em `C:\Users\Eu\voc-rag`), VS Code ou CMD.
- Hardware: 32 GB de RAM e 8 GB de VRAM. Tudo deve rodar localmente e de graca.
- Trabalhe em passos pequenos: um arquivo ou um problema por vez. Entregue arquivos completos, nao trechos soltos.
- Nunca invente dados, caminhos ou nomes de funcoes. Se faltar informacao, pergunte.
- Se for corrigir um erro, peca a mensagem de erro completa e o arquivo envolvido.
- Sempre lembre coisas simples: quando usar o Git para salvar o progresso (`git add`, `git commit`, `git push`, dentro de `C:\Users\Eu\voc-rag`) e os comandos do CMD que vem antes (`cd C:\Users\Eu\voc-rag`, `.venv\Scripts\activate`).
- Comandos Python usam `--` (dois hifens) nas opcoes. Streamlit ocupa a janela do CMD enquanto roda (Ctrl+C para parar).
- Nao diga que algo funciona sem ter visto o resultado real; separe "testei com simulacao" de "voce precisa testar".

## O projeto
Tech Challenge Fase 4 (POSTECH): sistema de Voice of Customer com RAG sobre as avaliacoes da base Olist (Brazilian E-Commerce, Kaggle).
**Prazo de entrega: 17/11/2026.**

Entregaveis: repositorio publico no GitHub (codigo, README completo com arquitetura, decisoes, exemplos, instrucoes de reproducao) e video de ate 7 minutos (problema, arquitetura, estrategia RAG, recuperacao, demonstracao, exemplos, evidencias, desafios e decisoes).

Requisitos principais:
1. Base de conhecimento justificada (selecao, limpeza, ausentes, documentos, embeddings, indexacao).
2. Pipeline RAG: transformacao da pergunta, recuperacao semantica, selecao, construcao do contexto, geracao.
3. Respostas fundamentadas, mostrando as evidencias (avaliacoes) usadas.
4. Tratar perguntas sem evidencia suficiente ou fora do escopo, sem inventar.
5. Opcional: metadados (nota, periodo, categoria, entrega) para filtros, com justificativa.

## Stack
- LLM de geracao: Ollama com `qwen3:8b` (http://localhost:11434), `think: false`, temperatura 0,1, `keep_alive` 30m.
- Embeddings: `intfloat/multilingual-e5-base` (prefixos `passage: ` e `query: `; vetores normalizados; 768 dims).
- Banco vetorial: ChromaDB local (`data/chroma_db`), colecao `avaliacoes_olist`, cosseno.
- Interface: Streamlit (`src/app.py`).
- Versoes (requirements.txt): chromadb 1.5.9, numpy 2.4.6, pandas 3.1.0, requests 2.34.2, sentence-transformers 6.2.0, torch 2.15.0, streamlit (sem versao fixada).
- Repositorio: github.com/Vangelo42/voc-rag-olist (`.gitignore` exclui `.venv/` e `data/`; `avaliacao/` e `docs/` sao versionados).
- Dica: `setx HF_HUB_OFFLINE 1` (e reabrir o CMD) evita consultas de rede ao Hugging Face; o modelo ja esta em cache.

## Estrutura de pastas
```
voc-rag\
  .venv\  data\ (CSVs Kaggle, processed\, chroma_db\)
  avaliacao\            (rodada 1: resultados.csv, citacoes_para_rotular.csv, resumo.txt)
    rodada2\            (idem; rodada 2 = a das CITACOES rotuladas)
    rodada3\            (rodada 3 = classificador final; numeros de comportamento e tempos do README)
  docs\img\             (capturas de tela para o README: resposta.png, filtros.png, fora_do_escopo.png)
  explorar.py
  README.md  requirements.txt  CONTEXTO.md
  src\
    config.py  preparar_base.py  enriquecer_metadados.py  explorar_csvs.py
    indexar.py  buscar_teste.py  calibrar_limiar.py
    filtros.py  rag.py  app.py  avaliar.py
```
Ordem para reconstruir: `preparar_base.py` -> `enriquecer_metadados.py` -> `indexar.py` (le reviews_enriquecidas.csv; usa o cache .npy dos embeddings se o tamanho bater).

## Decisoes da base (numeros reais)
- Funil: 99.224 -> 40.977 (com comentario, 41%) -> 40.950 -> 40.641 (review_id duplicado) -> **37.994** (min. 2 palavras e 3 letras). Titulo vazio em 88%.
- Um documento por avaliacao, sem chunking (mediana 60 caracteres, max. 229). Documento = titulo + comentario. Sem remover acentos/stopwords. Textos repetidos (3.592) mantidos.
- Viés: comentam 77% dos clientes de nota 1 e 36% de nota 5. Periodo: 2016-10 a 2018-08.
- Embeddings na CPU: 469 s. Calibracao do limiar (16 dentro, 14 fora): top-1 dentro min 0,845/mediana 0,896/max 0,941; fora min 0,787/mediana 0,839/max 0,850; regra: >= 3 textos unicos com sim >= 0,85 (94% das validas, 100% das fora).

## Metadados (Fase 4)
- Indice: `nota`, `faixa_nota`, `ano_mes`, `data_avaliacao`, `order_id`, `categoria`, `status_pedido`, `ano_mes_compra`, `faixa_entrega`, `n_categorias`, `dias_atraso` (so se entregue).
- `faixa_entrega`: compara DATAS (entrega ao cliente x estimada): no_prazo 32.538, atrasada 3.647, nao_entregue 1.809. Atraso medio -11 dias (chega antes da estimativa). Nota negativa: atrasada 74,1%, nao_entregue 86,5%, no_prazo 19,6% (justificativa dos metadados).
- Categoria: 73 + `sem_categoria` (1.049). Pedido com varias categorias (467, 1,2%): a mais frequente. Categorias sobrepostas (casa_conforto/_2, eletrodomesticos/_2, ferramentas_jardim/construcao_ferramentas_jardim) = limitacao.
- Periodo usa o mes da COMPRA (`ano_mes_compra`); a interface mostra compra e avaliacao.
- Filtros inferidos (`filtros.py`, regras, 21 autotestes, `python src/filtros.py`): "problemas/reclamacoes/insatisfeitos" -> negativa; "elogiam" -> positiva; "nota N"; "atraso" -> atrasada; "nao chegou" -> nao_entregue; anos/meses. Na duvida, sem filtro. Categoria: o LLM sugere (so entre as 73 reais), `validar_categoria` descarta invalidas. Filtros manuais mandam sobre os inferidos.

## Pipeline (`rag.py` v4)
1. Uma chamada do LLM: linha 1 `ESCOPO: DENTRO|FORA - motivo`, 3-5 frases no estilo de avaliacoes, `CATEGORIA: x|nenhuma`. Regra do prompt (versao de 14/11): ano, mes, nota e categoria NAO tiram a pergunta do escopo; FORA = assunto alheio ou pergunta cuja resposta e um NUMERO/dado da empresa; dois exemplos no prompt (um de cada lado). O motivo escrito pelo modelo nao e confiavel (copia o dos exemplos). O modelo as vezes responde so `FORA`; o parser (`saida_indica_fora_do_escopo`) olha so o inicio da saida. Saida bruta em `--debug` (`saida_reescrita`).
2. Busca de 30 candidatos por consulta (original + frases) com `where` dos filtros.
3. Limiar 0,85 + >= 3 textos unicos, senao `sem_evidencia` (com aviso se havia filtros).
4. Selecao de ate 10 evidencias por revezamento, ignorando repetidos e quase iguais (cosseno >= 0,93).
5. Prompt: so avaliacoes numeradas, citacoes [n] (max. 3), sem percentuais/frequencia/causas.
6. Guarda de frequencia: detecta ("principalmente", "maioria", "varios"...; ignora termos que estao na pergunta), pede uma nova versao (mantendo formato e citacoes por topico), senao aviso. Checagem de citacoes (existencia).
- CLI: `python src/rag.py "pergunta" --debug`; opcoes `--faixa --entrega --categoria --meses --sem-filtros --sem-reescrita --listar-categorias`.
- API: `RAG().perguntar(pergunta, filtros_manuais, reescrever, filtros_automaticos)`.
- Leitura de metadados do indice em lotes de 2.000 (limite de variaveis do SQLite).

## Avaliacao (Fase 5, `avaliar.py`)
Modos: `--executar`, `--rotular`, `--revisar [--rotulo nao]`, `--resumo`; `--rodada rodada2` escolhe a pasta (sem ela = rodada 1 em `avaliacao\`).
- **Rodada 1** (23 perguntas): validas 10/14 respondidas (V05, V07, V10, V12 recusadas por engano pelo classificador); fora do escopo 15/15 recusadas.
- Correcao: prompt do classificador (DENTRO explicito, "na duvida DENTRO", motivo curto) + perguntas novas V15-V20 e E06-E08 (nao usadas no ajuste).
- **Rodada 2** (32 perguntas, 48 execucoes): validas 20/20; fora do escopo 24/24 (8 perguntas x 3) todas pelo classificador; novas: validas 6/6, fora 9/9. Dificeis: D01 e D04 `fora_do_escopo` (mensagem imprecisa: sao do dominio, talvez sem evidencia); D02 e D03 respondidas.
- Tempos (rodada 2): mediana 7,5 s (media 8,3, max 12,1); reescrita 3,5, busca 0,6, geracao 3,5; recusa ~2,4 s. 1a chamada depois de ligar o PC ~26 s.
- Guardas (rodada 2, 22 respostas): evidencias por resposta 9,5; frequencia detectada em 5, nenhuma persistiu; 0 citacoes inexistentes; 0 respostas sem citacao.
- **Citacoes (rodada 2, 60 pares rotulados por 1 pessoa, apos revisao com criterio escrito)**: sustenta 27 (45%, IC 33-58%), parcial 29 (48%, 36-61%), nao 4 (7%, 3-16%); sustenta+parcial 56 (93%, 84-97%). Primeira passada (antes da revisao): 30%/48%/22% -> criterio sensivel ao avaliador. Numeros conferidos no resumo de 07/11 (60 de 165; 27/29/4).
- Falha conhecida D02: pergunta com premissa falsa ("elogios sobre entregas atrasadas") foi respondida repetindo a premissa, com citacoes de entregas no prazo.
- **Falha depois da rodada 2 (14/11):** "Quais reclamacoes sobre atraso na entrega apareceram em 2017?" (botao de exemplo da interface, fora da lista) foi recusada em 5/5 execucoes com o motivo "pedidos de contagens e filtros de datas". Corrigido no prompt do classificador (ver Pipeline).
- **Rodada 3** (40 perguntas, 62 execucoes; `avaliacao\rodada3\`): validas 25/25; fora do escopo 33/33 (11 perguntas x 3), todas pelo classificador; novas da rodada 3 (V22-V25, E09-E11, nao usadas no ajuste): validas 4/4, fora 9/9 execucoes; V21 (a que motivou o ajuste) respondida. Dificeis: D01, D02 e D03 respondidas, D04 `fora_do_escopo` (D01 mudou: respondida na 1, recusada na 2, respondida na 3). Tempos: mediana 8,1 s (media 8,8, max 13,2); reescrita 3,7, busca 0,4, geracao 3,8; recusa ~2,4 s. Guardas (28 respostas): 9,2 evidencias por resposta; frequencia detectada em 6, nenhuma persistiu; 0 citacoes inexistentes; 0 sem citacao. Citacoes NAO foram rotuladas de novo (prompt de geracao igual): vale a medida da rodada 2.
- Falha conhecida D01 (rodada 3): "aplicativo da loja" foi respondida com entrega rapida e satisfacao com produtos, sem citar aplicativo e sem avisar que o tema nao aparece: resposta fora do assunto (nao inventou, mas nao respondeu a pergunta). Ja esta nas Limitacoes do README.

## Problemas conhecidos / limitacoes (ja no README)
Citacoes so 45% sustentam por inteiro; premissa falsa; classificador LLM variavel e limiar fragil (0,849 vs 0,85 no caso da previsao do tempo); mensagem "fora do escopo" imprecisa para assuntos do dominio sem evidencia; "principais" = mais parecidas, nao mais frequentes; viés de quem comenta; categorias sobrepostas; avaliacao pequena e por 1 avaliador.

## Status
- [x] Fases 0 a 3 (Ollama, base, indexacao, pipeline v2).
- [x] Fase 4: v3 (guarda de frequencia), metadados (categoria, entrega, periodo), filtros inferidos, rag v4, interface Streamlit (testada com print em 07/11).
- [x] Fase 5: avaliacao rodadas 1, 2 e 3 (citacoes rotuladas na rodada 2: 27/29/4 de 60, conferido em 07/11).
- [x] README.md e requirements.txt escritos (07/11) e README atualizado com a rodada 3 (14/11); falta conferir as imagens no GitHub.
- [ ] Commit/push de tudo (`git add src/ avaliacao/ README.md requirements.txt CONTEXTO.md`; `docs/` so depois de criar `docs\img\`).
- [ ] Capturas de tela em `docs\img\` (resposta.png, filtros.png, fora_do_escopo.png) ou remover as 3 linhas de imagem do README.
- [ ] Fase 7: video de ate 7 min (roteiro: problema -> arquitetura -> pipeline -> demo com filtros e evidencias -> recusa -> avaliacao e limitacoes -> desafios/decisoes).
- [ ] Trilha Databricks (+10 pontos) em paralelo (comprovante pela Databricks Academy).
- [ ] Opcional (se sobrar tempo, ~12/11): rodada 3 com melhorias de prompt (max. 2 citacoes por afirmacao, usar palavras das avaliacoes, apontar premissas nao confirmadas) e comparar com a rodada 2.

## Proximo passo
1. Commit e push do rag.py, avaliar.py, avaliacao\rodada3, README e CONTEXTO de 14/11.
2. Fazer as capturas de tela; abrir o README no GitHub e conferir imagens, tabelas e o diagrama mermaid.
3. Roteiro e gravacao do video (ate 7 min); ensaiar com a interface aberta e o Ollama ligado.
4. Trilha Databricks.
5. Revisao final e envio com 1 dia de folga (alvo: 16/11).

## Registro de trocas (preencher a cada troca de assistente)
- Data / quem trabalhou / o que foi feito / o que ficou pendente:
- 02/11/2026 | Claude (chat original) | Fases 0 a 3 concluidas e testadas | Pendente: v3 do rag.py, Fase 4 em diante.
- 07/11/2026 | Claude (2o chat) | v3/v4 do rag.py, metadados, filtros, Streamlit, avaliacao (rodadas 1 e 2), README, requirements, CONTEXTO | Pendente: conferir numeros de citacoes, capturas, commit, video, Databricks.
- 14/11/2026 | Claude (2o chat) | Falha do classificador com periodo (botao de exemplo) corrigida; rodada 3 (25/25 e 33/33); README, CONTEXTO e roteiro atualizados | Pendente: capturas, commit, video (ate 7 min), Databricks; entrega 17/11.
