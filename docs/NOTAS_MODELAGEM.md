OBS:
- O texto que iniciar com [P] é uma pergunta/solicitação do usuário e deverá ser respondida com [R]. Cada pergunta poderá ter outras perguntas aninhadas. As perguntas deverão ser respondidas, respeitando o aninhamento.
- O texto que iniciar com [D] é uma definição/escolha de desenvolvimento. Caso você tenha algum ponto contrário ao que está definido, aponte.
- Existe a etapa 1_PRE_PROC, que é uma etapa inteira de pré-processamento e existe a etapa PRE_PROC_MODEL, que é uma sub-etapa dentro do TREINO/MODE_X.
- O documento atual tem como objetivo mapear os pontos do fluxo que será implementado nos notebooks 3_TREINO_MODE_C, 4_INFERENCIA_MODE_C, 5_COMP_MODE_C - que são notebooks a serem desenvolvidos - de modo a fornecer uma visão sobre como o código será desenvolvido, etapa por etapa. Contudo, as escolhas/implementações nos notebooks a serem desenvolvidos podem ter relação direta com demais etapas anteriores, portanto, você deverá considerar este aspecto na proposição de desenvolvimento. Caso haja a necessidade de modificar algum notebook de uma etapa anterior, proponha.
- Com base no documento atual NOTAS_MODELAGEM.md, a ideia é desenvolver um documento seguinte chamado PLANO_MODELAGEM.md, que deverá contemplar o plano conceitual de implementação/execução, contendo, em um nível de detalhe adequado:
    - Descrição do que será executado em cada etapa
    - Decisões técnicas definidas em cada etapa
    - Descrição, em alto nível, do código que será implementado, quando necessário
- Antes de criarmos o documento PLANO_MODELAGEM.md, leia o documento NOTAS_MODELAGEM.md (este documento) para validarmos o direcionamento que vamos seguir e o que será definido em PLANO_DEV.md
- Caso tenha faltado alguma anotação em alguma etapa específica, proponha adicionar.

---

- [P] Descreva o que será logado em cada etapa, nas etapas correspondentes deste fluxo
    - [R] Os logs específicos de cada etapa estão detalhados nas seções correspondentes abaixo. Resumo por notebook:
        - **3_TREINO_MODE_C / PRE_PROC_MODEL**: params de split/filtro (salt, frac, status permitidos, col names), FQNs de input/output, `n_linhas_por_regra` (contagem após cada regra), `rules_catalog.json`, profiling de df_model e df_validacao.
        - **3_TREINO_MODE_C / FEATURE_SELECTION**: params do FS (seeds, train_frac, thresholds, métodos), métricas de AUC-PR e AP por método/seed, ranking final de features, sets top-K, MI (`mi/mutual_information.csv/.json`), Pearson entre features selecionadas (`pearson/pearson_correlation.csv`, `pearson/pearson_heatmap.png`).
        - **3_TREINO_MODE_C / TREINO**: params do CV e do grid, CV metrics por fold/combo (auc_pr, avg, std), `model_ids` (JSON com mapeamento id → params), artefato de modelo e pipeline por combo (`treino_final/{model_id}/model`, `treino_final/{model_id}/preprocess_pipeline`), `cv/grid_results.json`, `cv/trained_models_registry.json`. Sem métricas de hold-out (delegadas ao 5_COMP).
        - **4_INFERENCIA_MODE_C**: params de referência (treino_run_id, model_ids inferidos, threshold por modelo, FQNs), n_input, n_output, profiling do output, score_profile por modelo.
        - **5_COMP_MODE_C**: params de análise (model_ids, K_LIST, CONVERSAO_LIST, LIFT_TARGET, baseline), métricas de performance por modelo (AUC-PR, AP, Precision@K, Recall@K, Lift@K, TP/FP/FN/TN @K), artefatos de gráficos comparativos e de overfitting. Detalhes na seção do 5_COMP.

# Fluxo notebook 3_TREINO_MODE_C
- [D] A mesma abordagem para SPLIT_SALT, CV_SEED e FS_SEEDS deverá ser implementada
    - [P] Descreva qual será a abordagem deste ponto e quais suas implicações
        - [R] Os três parâmetros controlam aleatoriedade/particionamento e devem ser sempre logados como params nas runs que os utilizam (conforme CLAUDE.md). Abordagem de cada um:
            - **SPLIT_SALT** (usado no PRE_PROC_MODEL): string concatenada ao vetor `(ID_COL, SEG_COL, MES)` antes do `xxhash64`. Determina qual registro vai para treino ou validação de forma determinística e estratificada por segmento e mês. Implicação: alterar o salt gera uma partição completamente diferente, incompatível com execuções anteriores. O mesmo salt precisa ser usado em qualquer re-execução que precise reproduzir a partição. A partição fica materializada em tabela (df_model, df_validacao), então a inferência não precisa replicar o salt — mas o salt deve constar no log como rastreabilidade da partição.
            - **CV_SEED** (usado no TREINO): semente do `xxhash64` na função `kfold_split`, que define quais registros caem em cada fold. Implicação: mudar CV_SEED altera os folds e, portanto, os valores de AUC-PR por fold. A comparação entre combos do grid só é válida quando todos usam o mesmo CV_SEED.
            - **FS_SEEDS** (usado no FEATURE_SELECTION): lista de seeds para o loop multi-seed no FS. Para cada seed, o FS faz `sampleBy` estratificado (por label), treina LR-L1, RF e GBT, e computa importâncias. O ensemble final é a média ponderada por AUC-PR. Implicação: mais seeds aumentam a estabilidade do ranking de features, mas aumentam o custo computacional (seeds × métodos × fit + predict). A lista de seeds deve ser idêntica entre re-execuções que pretendam comparar resultados de FS.
- [D] Quero escolher Configs de forma livre, sem comprometer as lógicas do código. Por exemplo, caso eu queira treinar com um grid menor/maior, posso definir sem quebrar o resto do notebook. Isso vale para o restante dos parâmetros que serão definidos na configuração da execução. Configs alteram os inputs das funções, não a lógica interna.
- [D] Quero definir as configs etapa por etapa (por exemplo, no notebook 3_TREINO_MODE_C: PRE_PROC_MODEL > FEATURE_SELECTION > TREINO), se possível. Algumas das configs são mapeadas a seguir, embora possam estar faltando as demais:
    - Gerais
        - MLflow (estrutura), Versionamento, INPUT, OUTPUT
        - `SPLIT_SALT`, `CV_SEED`, `FS_SEEDS` definidos aqui
    - Pré-Processamento (PRE_PROC_MODEL)
        - PRE_PROC_MODEL — params (`STATUS_COL`, `LABEL_COL`, `ID_COL`, `SEG_COL`, `DATE_COL`, `VALID_FRAC`, `SPLIT_SALT`)
        - Regras de processamento definidas (Regras, Toggles, Catálogo) — mesma engine `rule_def` + `apply_rules_block`
        - Definição de FS_DECIMAL_COLS, FS_DIAS_COLS, FS_CAT_COLS: ler `df_seg.columns` logo após carregar a tabela → exibir lista → definir as três listas a partir das colunas disponíveis → usar um set `EXCLUIR_DE_FEATURES` com exclusões explícitas e documentadas. Isso evita enviar colunas que não existem e torna a escolha auditável.
        - Thresholds de drop de colunas com nulos e constantes, e de truncagem de cardinalidade, assim como 'SEG_TARGET' deverão estar definidos em Pré-Processamento.
    - Feature Selection (FEATURE_SELECTION)
        - FS — params: `FS_SEEDS`, `FS_TRAIN_FRAC`, `TOPK_LIST`, listas de features a serem alimentadas ao FS
    - Treino (TREINO)
        - `ID_COLS` e `DROP_FROM_FEATURES` — mesma lógica do MODE_B: `ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]`; `DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]`
        - Pipeline de classificação: GBT + CV determinístico por hash — manter arquitetura do MODE_B (ver resposta detalhada abaixo)
- [D] Faça uma lista e descrição das configs que serão definidas
    - **Gerais**
        - `EXPERIMENT_NAME`: path do experimento MLflow
        - `PR_TREINO_NAME`: nome do parent run container do treino (ex: `T_PR_TREINO`)
        - `MODE_CODE`: código do mode (ex: `"C"`)
        - `PR_RUN_ID_OVERRIDE`, `MODE_RUN_ID_OVERRIDE`, `PRE_PROC_RUN_ID_OVERRIDE`, `FS_RUN_ID_OVERRIDE`, `TREINO_RUN_ID_OVERRIDE`: overrides para reutilizar runs existentes
        - `TREINO_VERSAO`: versão do experimento (ex: `"V1"`)
        - `COTACAO_SEG_FQN`: FQN da tabela silver de entrada
        - `OUT_SCHEMA`: schema de saída (ex: `"gold"`)
        - `SPLIT_SALT`: string de salt para particionamento determinístico
        - `CV_SEED`: seed do k-fold CV
        - `FS_SEEDS`: lista de seeds para o FS multi-seed
    - **PRE_PROC_MODEL**
        - `STATUS_COL`, `LABEL_COL`, `ID_COL`, `SEG_COL`, `DATE_COL`: nomes das colunas estruturais
        - `ALLOWED_FINAL_STATUS`: lista de status finais permitidos (ex: `["Emitida", "Perdida"]`)
        - `VALID_FRAC`: fração de validação (ex: `0.20`)
        - `SEG_TARGET`: segmento alvo para filtro
        - `DO_PROFILE`: bool para ativar profiling
        - `FS_DECIMAL_COLS`, `FS_DIAS_COLS`, `FS_CAT_COLS`: listas de features candidatas por tipo
        - `EXCLUIR_DE_FEATURES`: set de colunas a excluir explicitamente das features candidatas
        - `NULL_DROP_PCT`: threshold de fração de nulos para drop de coluna (ex: `0.90`)
        - `HIGH_CARD_THRESHOLD`: threshold de cardinalidade para truncagem (ex: `15`)
        - `HIGH_CARD_TOP_N`: número de categorias top a manter (ex: `10`)
        - `OUTROS_LABEL`: label para categorias truncadas (ex: `"OUTROS"`)
        - Toggles e catálogo de regras (mesma engine do MODE_B)
    - **FEATURE_SELECTION**
        - `FS_SEEDS`: já definido em Gerais
        - `FS_TRAIN_FRAC`: fração de treino no sampleBy do FS (ex: `0.70`)
        - `TOPK_LIST`: lista de K's para geração de feature sets (ex: `[5, 7, 12]`)
        - `FS_METHODS_CONFIG`: dict com configuração de cada método (LR-L1, RF, GBT)
    - **TREINO**
        - `TREINO_FEATURE_SET_KEY`: chave do feature set selecionado do FS (ex: `"top_7"`)
        - `USE_CLASS_WEIGHT`: `"auto"` | `True` | `False`
        - `CLASS_WEIGHT_THRESHOLD`: threshold de label_rate para ativar class weight (ex: `0.30`)
        - `CV_FOLDS`: número de folds do CV (ex: `3`)
        - `CV_METRIC`: métrica de avaliação do CV (ex: `"areaUnderPR"`)
        - `GBT_PARAM_GRID`: dict com listas de valores para cada hiperparâmetro e valor fixo de `maxIter`. Exemplo:
            ```
            GBT_PARAM_GRID = {
                "maxDepth": [4, 6],
                "stepSize": [0.05, 0.1],
                "maxIter":  100,
            }
            ```
            Os pares são gerados via `itertools.product` das listas. Alterar as listas muda o grid sem impactar a lógica.
        - `ID_COLS`, `DROP_FROM_FEATURES`: mesmos do MODE_B
- [D] Os logs, a princípio se mantém iguais, mas quero logar lineage da etapa de PRE_PROC_MODEL → adicionar ao log da run PRE_PROC_MODEL: `input_cotacao_seg_fqn`, `n_linhas_por_regra` (contagem após cada regra aplicada), e o catálogo de regras (já feito em MODE_B via `rules_catalog.json`). Isso constitui o lineage da etapa.

## Definições etapa Pré-Processamento (PRE_PROC_MODEL)
- [D] Regras aplicadas deverão ser específicas de preparação para ML e devem permanecer apenas em 3_TREINO_MODE_C, por exemplo: Encoding → Imputer (média) → VectorAssembler — apenas em 3 (pipeline Spark ML, fit no treino, transform no treino e validação). Demais regras já foram aplicadas no notebook 1_PRE_PROC.
- [D] A etapa de PRE_PROC_MODEL consome da tabela silver (`COTACAO_SEG_FQN`) e produz as tabelas gold `df_model` e `df_validacao`. É possível adicionar transformações específicas da transição silver → gold via a mesma engine (`rule_def` + `apply_rules_block` + toggles + catálogo). Estas são as únicas regras adequadas para esta etapa — transformações de preparação para ML (encoding, imputer, assembler) pertencem ao TREINO.
    - [P] Em PRE_PROC_MODEL, qual a diferença entre os prefixos `PP_` e `BUILD_` nas regras?
        - [R] Os prefixos organizam semanticamente o propósito de cada regra:
            - **`PP_`** (Pre-Processing): operações sobre dados já existentes na silver — normalização, filtragem, criação de rótulo. Não criam colunas auxiliares derivadas de lógica de particionamento. Ex: `PP_R01_normaliza_status`, `PP_R02_filtra_status_finais`, `PP_R03_cria_label`.
            - **`BUILD_`**: construção de colunas auxiliares necessárias para o pipeline downstream. Colunas que não existem na silver e são derivadas aqui. Ex: `BUILD_R01_add_mes` (coluna `MES` derivada de `DATE_COL`), `BUILD_R02_add_split_flag` (flag determinística treino/validação via hash). São pré-requisitos para o split.
            - **`MODEL_` / `VALID_`**: filtros finais que materializam o split em dois DataFrames distintos (`df_model`, `df_validacao`), aplicados após `BUILD_R02`. Não são toggleáveis — devem sempre estar habilitadas.
            - A separação mantém o catálogo legível e permite ativar/desativar grupos de regras independentemente.
- **Logs desta etapa** (run exec T_PRE_PROC_MODEL):
    - Tags: `pipeline_tipo`, `stage`, `run_role`, `mode`, `step`, `treino_versao`, `versao_ref`
    - Params: `ts_exec`, `treino_versao`, `mode_code`, `seg_target`, `input_cotacao_seg_fqn`, `df_model_fqn`, `df_validacao_fqn`, `valid_frac`, `split_salt`, `allowed_final_status`, `label_col`, `id_col`, `seg_col`, `pr_run_id`, `mode_run_id`
    - Metrics: `n_seg_in`, `n_seg_after_rules`, `n_linhas_por_regra_{rule_id}` (uma métrica por regra aplicada, contagem de linhas após cada regra), `n_df_model`, `n_df_validacao`
    - Artifacts: `rules_catalog.json`, `rules_execution.json`, `profiling_df_model.json`, `profiling_df_validacao.json`, `eda_df_model_by_seg.json`

## Definições etapa Feature Selection (FEATURE_SELECTION)
- [D] A etapa deverá ser implementada com base no que foi feito para 3_TREINO_MODE_B.
- [D] Deverá ser implementada também análise de MI (Mutual Information), apenas para análise, de modo a complementar os algoritmos implementados (RF, GBT, LR). MI não entrará como seletor
- [D] Deverá ser implementada análise de Pearson (Correlação) entre variáveis selecionadas após Feature Selection
- [P] Descreva como esta etapa será implementada e como serão feitos os logs
    - [R] A implementação segue o MODE_B como base (loop multi-seed com LR-L1, RF, GBT → ensemble ponderado por AUC-PR → top-K sets). As adições em MODE_C são:
        - **MI**: já implementado no MODE_B via `mutual_info_classif` (sklearn). Sample de até 50k linhas, encoding label de categóricas, imputação por mediana. Resultado logado como `mi/mutual_information.csv` e `mi/mutual_information.json`. Mantido em MODE_C sem alteração.
        - **Pearson**: calculado *após* o FS, sobre as features numéricas (`NUM_COLS_FINAL`) presentes em `FS_FEATURES_RANKED` (todas as features rankeadas, não apenas o top-K). Implementação: coletar amostra para pandas (mesma amostra do MI ou nova, até 50k linhas), calcular `df[num_cols].corr(method="pearson")`. Logar como:
            - `pearson/pearson_correlation.csv`: matrix de correlação em formato longo (feature_a, feature_b, correlation)
            - `pearson/pearson_heatmap.png`: heatmap matplotlib, anotado com valores, features ordenadas por rank do FS
        - Pearson não afeta o ensemble nem a seleção de features — é análise exploratória logada apenas como artefato.
    - **Logs desta etapa** (run exec T_FEATURE_SELECTION):
        - Tags: `pipeline_tipo`, `stage`, `run_role`, `mode`, `step`, `treino_versao`, `versao_ref`
        - Params: `df_model_fqn`, `seg_target`, `fs_seeds`, `fs_train_frac`, `null_drop_pct`, `high_card_threshold`, `high_card_top_n`, `outros_label`, `fs_methods`, `fs_methods_config`, `topk_list`, `fs_decimal_cols`, `fs_dias_cols`, `fs_cat_cols`, `ensemble_type`, `mi_in_ensemble=false`, `pr_run_id`, `mode_run_id`, `fs_container_run_id`
        - Metrics: `n_rows_seg`, `n_label_invalid_or_null`, `n_features_candidate`, `{method}_seed{seed}_ap_val`, `{method}_seed{seed}_auc_pr_val`, `{method}_avg_ap_val`, `{method}_avg_auc_pr_val`
        - Artifacts: `fs_stage1/null_profile.json`, `fs_stage1/cat_cardinality.json`, `fs_stage1/fs_feature_contract.json`, `methods/{method}/seed{seed}/importance_by_feature.csv`, `methods/{method}/importance_avg.csv`, `summary/ensemble_weights.json`, `summary/feature_ranking_final.csv`, `summary/features_ranked.json`, `summary/topk_sets.json`, `mi/mutual_information.csv`, `mi/mutual_information.json`, `pearson/pearson_correlation.csv` (novo), `pearson/pearson_heatmap.png` (novo)

## Definições etapa Treinamento (TREINO)
- [P] Após o treinamento do modelo, quais as formas de cálculo e definição do threshold de classificação (threshold que define o score de corte para classificação como 0,1)? Explique de que forma este threshold de classificação implica demais análises/resultados/performances.
    - [R] Existem quatro abordagens principais para definição do threshold:
        1. **Threshold por capacidade operacional** (abordagem do MODE_B): `K = int(n_hold_out × CAPACIDADE_PCT)` → threshold = score do K-ésimo elemento no ranking descendente por score. É o mais alinhado com o problema de negócio (capacidade do time). Derivado do K, não do threshold em si.
        2. **Threshold por ponto ótimo na curva PR** (F1 máximo): varrer todos os thresholds candidatos (scores únicos) e escolher o que maximiza F1 = 2×precision×recall/(precision+recall). Mais geral, mas ignora a restrição de capacidade.
        3. **Threshold por recall mínimo** (minimizar FN): escolher o threshold mais alto que ainda mantém recall ≥ valor definido. Adequado dado o foco em minimizar FN.
        4. **Threshold manual**: fixado diretamente pelo usuário (ex: `0.5`).
        - **Implicações do threshold sobre demais análises**:
            - O threshold define o ponto de operação na curva PR → determina precision e recall de operação.
            - Confusion matrix (TP, FP, FN, TN) é função direta do threshold escolhido.
            - Precision@K e Recall@K: quando K é derivado da capacidade, o threshold operacional é o score do K-ésimo elemento — equivalente ao threshold de capacidade.
            - Lift@K = Precision@K / baseline: depende do threshold de capacidade.
            - AUC-PR e AP são independentes do threshold (métricas de ranking global).
            - Mudar o threshold não altera a curva PR, apenas o ponto marcado sobre ela.
        - [P] Quero definir o threshold de classificação utilizando o F1 máximo, embora eu esteja fazendo análises de capacidade. Como isso funcionaria? Faz sentido?
            - [R] Faz sentido, mas são dois critérios com objetivos distintos que coexistem sem conflito:
                - **Threshold por F1 máximo**: encontra o ponto da curva PR que maximiza `F1 = 2 × precision × recall / (precision + recall)`. É um critério global, agnóstico à capacidade do time. Resulta em um threshold fixo que equilibra precision e recall da melhor forma possível sobre o hold-out.
                - **Threshold por capacidade (K)**: derivado de `K = int(n_hold_out × CAPACIDADE_PCT)` → threshold = score do K-ésimo elemento. É um critério operacional — define o corte em função do volume que o time consegue atender, não de uma métrica de modelo.
                - **Como coexistem**: os dois thresholds marcam pontos *diferentes* sobre a mesma curva PR. É possível calcular ambos para cada modelo e comparar: se o threshold de F1 máximo cai próximo ao threshold de capacidade, os dois critérios convergem. Se diferem muito, significa que a capacidade do time está operando longe do ponto ótimo de F1.
                - **Implementação prática no 5_COMP_MODE_C**: calcular ambos os thresholds por modelo e reportar lado a lado. Para o threshold de F1 máximo: varrer os scores únicos do hold-out como candidatos a threshold, calcular TP/FP/FN para cada um e selecionar o que maximiza F1. Pode ser feito com `df_ranked` já computado (window function sobre scores). Logar `threshold_f1_max_{model_id}` e `f1_max_{model_id}` como métricas no MLflow. Plotar ambos os thresholds marcados sobre a curva PR.
        - [P] Em qual etapa o threshold de classificação é definido? `pred_emitida` é gerada na mesma etapa?
            - [R] Em MODE_C, o threshold de classificação **não é definido no 3_TREINO** — é calculado no **5_COMP_MODE_C**, onde estão disponíveis os scores do hold-out e os parâmetros de negócio (`CAPACIDADE_PCT`, `LIFT_TARGET`). Dois thresholds distintos são calculados:
                - **`threshold_f1_max`**: maximiza F1 sobre os scores do hold-out (critério de modelo)
                - **`threshold_capacidade`**: score do K-ésimo elemento, derivado de `CAPACIDADE_PCT × n_hold_out` (critério operacional)
                - **`pred_emitida` = f(threshold)**: sim, `pred_emitida` é uma função direta do threshold escolhido. Portanto ela é gerada no **5_COMP**, no momento em que os thresholds são calculados e aplicados sobre `p_emitida`. O 4_INFERENCIA_MODE_C produz apenas `p_emitida` (score contínuo) — sem `pred_emitida`.
                - No COMP existem duas versões: `pred_emitida_f1` (usando `threshold_f1_max`) e `pred_emitida_k` (usando `threshold_capacidade`). Ambas derivadas do mesmo score, usadas para as respectivas confusion matrices e métricas @K.
                - **Distinção importante**:
                    - *Threshold de classificação*: ponto de corte do score para gerar rótulo binário (0/1). Critério de modelo — pode ser F1 máximo, recall mínimo, ou manual.
                    - *Threshold operacional (de capacidade)*: derivado de K — não é parâmetro de modelo, é consequência da capacidade do time. Coincide com o threshold de classificação quando o critério de operação é a capacidade.
- [D] Arquitetura do treino: GBT com CV 3-fold determinístico (hash por ID + CV_SEED) + grid search manual (4 combinações: maxDepth ∈ {4,6} × stepSize ∈ {0.05, 0.1}, maxIter fixo em 100). No entanto, como mencionado anteriormente, quero poder definir nas Configs quais os valores do grid e qual o tamanho do grid. A ideia é poder modifcar estes valores e quais parâmetros variar sem fazer alterações adiante no código
- [D] Mudança em relação ao MODE_B: sem seleção automática de vencedor. Cada combinação do grid gera um modelo salvo individualmente. Cada modelo receberá um id, de modo que, em etapas posteriores, cada modelo possa ser referenciado, como na hora de escolher para quais modelos vou gerar as análises de performance e resultados. A cada execução do notebook, para um grid específico, quero gerar apenas uma run que treina e armazena todos os modelos, de modo a serem referenciados posteriormente.
    - [P] Qual será a sugestão de implementação deste critério? Descreva.
        - [R] Cada combinação do grid gera:
            1. **ID único por modelo**: `model_id = f"d{maxDepth}_s{str(stepSize).replace('.','')}"` (ex: `d4_s005`, `d6_s01`). Gerado automaticamente a partir de `GBT_PARAM_GRID` via `itertools.product`, sem necessidade de edição manual.
            2. **Treino final completo** em `df_model_ml` (igual ao MODE_B, mas para cada combo, não apenas o vencedor).
            3. **Artefatos por modelo**:
                - Modelo GBT: `treino_final/{model_id}/model` (via `mlflow.spark.log_model`)
                - Pipeline de pré-processamento: `treino_final/{model_id}/preprocess_pipeline` (via `mlflow.spark.log_model`)
            4. **Dicionário interno** `TRAINED_MODELS = {model_id: {"params": combo, "cv_avg_auc_pr": ..., "cv_std_auc_pr": ...}}` mantido em memória durante o notebook e serializado como artefato `cv/trained_models_registry.json`.
            5. **Param `model_ids`** logado como JSON list na run exec do TREINO: `["d4_s005", "d4_s01", "d6_s005", "d6_s01"]`. Este param é o contrato entre o 3_TREINO e os notebooks 4 e 5.
            - A run do TREINO contém todos os modelos como artefatos. Para referenciar um modelo específico na inferência, basta fornecer o `TREINO_EXEC_RUN_ID` e o `model_id` desejado (ou uma lista de model_ids).
            - **Persistência entre sessões**: modelos e pipelines são salvos no artifact store do MLflow via `mlflow.spark.log_model`. Após encerramento do cluster, qualquer run posterior carrega o modelo com `mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model")` e o pipeline com `mlflow.spark.load_model(f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline")`. O par `(TREINO_EXEC_RUN_ID, model_id)` é a referência completa e suficiente. Em MODE_C, diferente do MODE_B, o pipeline **não precisa ser reconstruído** no 4_INFERENCIA — pode ser carregado diretamente do MLflow, eliminando a dependência de `DF_MODEL_FQN` na inferência.
- **Logs desta etapa** (run exec T_TREINO):
    - Tags: `pipeline_tipo`, `stage`, `run_role`, `mode`, `step`, `treino_versao`, `versao_ref`
    - Params: `df_model_fqn`, `df_valid_fqn` (referência, não usado para hold-out aqui), `seg_target`, `feature_set`, `feature_cols`, `n_features`, `n_model`, `use_class_weight`, `apply_cw`, `weight_pos`, `label_rate`, `cv_folds`, `cv_seed`, `cv_metric`, `gbt_param_grid`, `model_ids`, `mode_code`, `pr_run_id`, `mode_run_id`, `treino_container_run_id`
    - Metrics: `cv_{model_id}_fold{i}_auc_pr`, `cv_{model_id}_avg_auc_pr`, `cv_{model_id}_std_auc_pr` (para cada combo/fold)
    - Artifacts: `cv/grid_results.json`, `cv/fold_metrics.json`, `cv/trained_models_registry.json`, `treino_final/{model_id}/model` (um por combo), `treino_final/{model_id}/preprocess_pipeline` (um por combo)

# Fluxo notebook 4_INFERENCIA_MODE_C
- [D] A inferência nos dados de validação ainda retornaria apenas uma tabela, mas com as p–emitidas e demais informações derivadas da inferência, para cada modelo da run, gerado a partir do grid. Portanto, a partir desta tabela gerada, como mencionado, na etapa 5_COMP_MODE_C eu quero fazer a comparação entre estes modelos da mesma run e gerar análises para entender o comportamento dos modelos e selecionar um modelo vencedor, caso eu queira usá-lo posteriormente à etapa de comparação e análise.
- [P] A estrutura da tabela de output seria wide format (uma linha por cotação, colunas `p_emitida_{model_id}` por modelo) ou long format (uma linha por cotação × modelo)? Qual o problema do wide format?
    - [R] Ambas funcionam, mas o **wide format** tem os seguintes problemas em Spark:
        1. **Schema dinâmico**: os nomes de coluna dependem de `MODEL_IDS`, que varia entre execuções. Exige construção dinâmica de expressões e pode gerar colunas órfãs se o grid mudar.
        2. **Ranking por modelo**: calcular `rank_global` por score precisa ser feito separadamente para cada coluna `p_emitida_{mid}`, sem poder usar `partitionBy("model_id")`.
        3. **Análise no 5_COMP**: calcular AUC-PR, AP, métricas @K por modelo exige iterar sobre nomes de coluna ao invés de filtrar por `model_id`. O código do COMP fica acoplado à lista de `model_ids`.
        4. Vantagem do wide: uma linha por cotação — mais intuitivo para inspeção direta.
        - **Decisão**: **long format** (uma linha por cotação × modelo). Filtrar por `model_id` é idiomático em Spark e independe do número de modelos.
- **Estrutura da tabela de output** (long format): uma linha por cotação × modelo. Colunas: `{ID_COL}`, `{STATUS_COL}`, `{LABEL_COL}`, `{SEG_COL}`, `{DATE_COL}`, `model_id`, `p_emitida`, `rank_global` (rank descendente por score, dentro do modelo), metadados (`treino_exec_run_id`, `inf_versao`, `mode_code`, `seg_inferida`, `inference_ts`).
- **Logs desta etapa** (run exec T_INFERENCIA):
    - Tags: `pipeline_tipo`, `stage=INFERENCIA`, `run_role=exec`, `mode`, `inf_versao`
    - Params: `treino_exec_run_id`, `model_ids_inferred`, `input_table_fqn`, `output_table_fqn`, `seg_target`, `threshold_mode`, `df_model_fqn` (para reconstrução do pipeline)
    - Metrics: `n_input`, `n_output_por_modelo`, `n_null_p_emitida_por_modelo`
    - Artifacts: `profiling_output.json`, `score_profile_per_model.json`

# Fluxo notebook 5_COMP_MODE_C
- [D] Todas as análises de desempenho do 5_COMP_MODE_C são realizadas sobre os dados de **validação (hold-out)** — `df_validacao`, que não participou do treinamento. A tabela de input do COMP é o output do 4_INFERENCIA_MODE_C, produzido pela aplicação dos modelos sobre `df_validacao`.
    - [P] Como o isolamento do hold-out está garantido no fluxo?
        - [R] O isolamento está garantido pela estrutura do PRE_PROC_MODEL: `BUILD_R02_add_split_flag` cria a flag `is_valid` via hash determinístico sobre `(ID_COL, SEG_COL, MES, SPLIT_SALT)`. As regras `MODEL_R03` e `VALID_R03` materializam o split em duas tabelas gold distintas. O 3_TREINO usa **apenas** `df_model` no fit (CV + treino final). O 4_INFERENCIA aplica os modelos sobre `df_validacao` (configurado como `INPUT_TABLE_FQN`). O 5_COMP lê a saída do 4_INFERENCIA — portanto opera exclusivamente sobre exemplos que os modelos nunca viram. O `df_model` não é acessado diretamente no COMP; as métricas de treino (`auc_pr_treino_{model_id}`) já foram logadas no MLflow pelo 3_TREINO e são lidas via `MlflowClient`.
- [D] Nesta etapa, preciso definir os parâmetros que serão utilizados nas análises, como CAPACIDADE_PCT, LIFT_TARGET
    - [P] Faz sentido definir estes parâmetros apenas nesta etapa?
        - [R] Sim. Em MODE_C, o treino não calcula threshold operacional nem faz escolha de vencedor — tudo é delegado ao COMP. Portanto, `CAPACIDADE_PCT` e `LIFT_TARGET` são estritamente parâmetros de análise e devem residir apenas no 5_COMP_MODE_C. Ressalva: o 4_INFERENCIA_MODE_C pode precisar de um threshold para gerar a coluna `pred_emitida`. Opções: (a) usar threshold fixo padrão (ex: 0.5) apenas para a coluna binária, documentando que não é o threshold operacional; ou (b) não gerar `pred_emitida` na inferência e calculá-la dinamicamente no COMP com o threshold derivado de CAPACIDADE_PCT. A opção (b) é mais limpa e evita ambiguidade.
    - [P] Quais demais parâmetros poderiam/deveriam ser definidos nesta etapa para gerar as análises?
        - [R]
            - `TREINO_EXEC_RUN_ID`: run_id do T_TREINO de referência (para leitura dos model_ids e recuperação de métricas de CV)
            - `INFERENCIA_TABLE_FQN`: FQN da tabela gerada pelo 4_INFERENCIA
            - `MODEL_IDS`: lista dos model_ids a comparar — pode ser subconjunto do grid (permite focar em modelos de interesse)
            - `K_LIST`: lista de capacidades operacionais para análise (ex: `[0.05, 0.10, 0.15, 0.20]`). `CAPACIDADE_PCT` principal é um desses valores.
            - `CONVERSAO_LIST`: lista de taxas de conversão base para cálculo de lift em diferentes cenários (ex: `[0.10, 0.20, 0.30]`)
            - `LIFT_TARGET`: lift mínimo desejado — exibido como linha horizontal nos gráficos de Precision@K
            - `BASELINE_MODE`: `"taxa_base"` | `"conversao_time"`
            - `CONVERSAO_TIME`: valor float, usado se `BASELINE_MODE="conversao_time"`
            - `ATTR_ANALYSIS_COLS`: lista de colunas para análise de performance por dimensão (ex: `["DS_PRODUTO_NOME"]`)
            - `LABEL_COL`, `STATUS_COL`, `SEG_COL`, `DATE_COL`, `ID_COL`: colunas estruturais esperadas na tabela de inferência
- [D] Estas análises deverão ser construídas para comparar os diferentes modelos do mesmo grid de treino
- [D] Algumas das análises serão exploratórias e não necessariamente deverão ser logadas no MLflow direto, ficando apenas a nível de notebook e exploração. Faça uma descrição de quais análises seriam logadas no MLflow direto (análises mais amplas) e quais ficariam apenas a nível de notebook (análises exploratórias).
    - **Logadas no MLflow**:
        - Métricas de performance por modelo: `auc_pr_{model_id}`, `ap_{model_id}`, `precision_at_k_{model_id}`, `recall_at_k_{model_id}`, `lift_at_k_{model_id}` para CAPACIDADE_PCT principal
        - Confusion matrix @K por modelo: `tp_at_k_{model_id}`, `fp_at_k_{model_id}`, `fn_at_k_{model_id}` (e TN se incluído)
        - Overfitting gap por modelo: `auc_pr_treino_{model_id}`, `gap_auc_pr_{model_id}`
        - Artifacts: `comparativo/metrics_summary.json`, `comparativo/topk_curves_{metric}.png` (Precision@K, Recall@K, Lift@K por modelo sobrepostos), `comparativo/pr_curves.png` (curvas PR por modelo sobrepostas), `overfitting/overfitting_summary.json`, `overfitting/pr_curves_treino_vs_val_{model_id}.png`, `overfitting/score_distributions.png`
    - **Apenas no notebook (exploratório)**:
        - Análise de performance por atributo (ex: Precision@K para cada `DS_PRODUTO_NOME`)
        - Curvas @K para combinações de `CONVERSAO_LIST` × modelo (além da principal)
        - Distribuição de scores por mês/segmento
        - Evolução mensal de métricas
- [D] Quero gerar as análises sob três pontos de vista:
    - Análise de overfitting
        - [D] Para GBT (e qualquer modelo do grid): comparar métricas **treino vs validação**: AUC-PR, AP e Lift@K. Se treino >> validação, overfitting. Abordagens:
            - **Gap de AUC-PR**: diferença > 0.05 entre treino e val é sinal de overfitting. Logar ambos.
            - **Curva PR treino vs validação sobrepostas**: gap visual entre as curvas.
            - **Distribuição de scores**: se a distribuição no treino for muito mais concentrada nos extremos (0 e 1) do que na validação, o modelo memorizou.
            - Para GBT especificamente: overfitting manifesta via `maxDepth` alto ou `stepSize` alto — o grid contempla isso ao incluir `maxDepth ∈ {4, 6}` e `stepSize ∈ {0.05, 0.1}`.
            - [P] Descreva como estas análises serão logadas no MLflow
                - [R] Para calcular métricas de treino, o 5_COMP precisa aplicar cada modelo sobre `df_model_ml`. Duas abordagens:
                    - **Abordagem A (recomendada)**: logar AUC-PR e AP de treino já no 3_TREINO_MODE_C (após treino final no df_model), como `auc_pr_treino_{model_id}` e `ap_treino_{model_id}`. O 5_COMP lê esses valores via `MlflowClient.get_run(TREINO_EXEC_RUN_ID).data.metrics`. Evita reprocessamento no COMP.
                    - **Abordagem B**: reconstruir e aplicar o modelo no 5_COMP sobre uma amostra do df_model. Mais pesada, mas não requer alteração no 3_TREINO.
                - Preferencialmente a Abordagem A, adicionando ao 3_TREINO o cálculo de `auc_pr_treino` e `ap_treino` por modelo após o treino final (antes de fechar a run).
                - **Logs de overfitting no MLflow (run exec T_COMP)**:
                    - Metrics: `auc_pr_treino_{model_id}`, `auc_pr_val_{model_id}`, `gap_auc_pr_{model_id}` (para cada modelo)
                    - Artifacts: `overfitting/overfitting_summary.json` (tabela com gap por modelo), `overfitting/pr_curves_treino_vs_val_{model_id}.png` (um por modelo), `overfitting/score_distributions.png` (distribuições sobrepostas, treino vs val, por modelo)
        - [D] Análise de desempenho dos modelos
            - Precision@K, Recall@K, Lift@K
                - [D] No notebook 5_COMP_MODE_C, quero definir um LIFT_TARGET, K_LIST e CONVERSAO_LIST. Estes dois últimos serão utilizados para incluir nas análises. Por exemplo, gerar um gráfico de Precision@K, para diferentes valores de CONVERSAO_LIST, tendo um LIFT_TARGET mostrado como linha horizontal, para diferentes K_LIST, que indicam a capacidade do time. A ideia é poder verificar em quais cenários os modelos superariam a performance mínima
                - Quero definir uma lista de atributos (colunas da tabela usada para as análises), de modo que eu possa gerar, além da análise generalizada, uma análise de performance específica por atributo. Por exemplo, Precision@K para DS_PRODUTO_NOME, onde cada curva representará um produto. Esta análise fica apenas a nível de notebook e exploração
        - AUC_PR, Average Precision
        - Curva PR
            - [P] Como seria gerada esta análise? Como se relaciona com K's, LIFT_TARGET e demais parâmetros/variáveis?
                - [R] A curva PR é gerada varrendo todos os thresholds únicos (scores únicos da coluna `p_emitida`) e calculando precision/recall em cada ponto. Em Spark: via Window functions (row_number ordenado por score desc + cumsum de labels). Ou via `BinaryClassificationMetrics` do MLlib para AUC, com a curva coletada em pandas para plotagem.
                    - **Relação com K's**: K não muda a curva PR (que é contínua). Para cada K em `K_LIST`, marca-se um ponto sobre a curva correspondente ao threshold de capacidade daquele K. O ponto mostra onde o modelo opera com aquela capacidade.
                    - **Relação com LIFT_TARGET**: `precision_target = LIFT_TARGET × baseline`. Traça-se uma linha horizontal em `precision = precision_target` no gráfico da curva PR. Os pontos onde a curva cruza essa linha indicam os intervalos de recall em que o modelo supera o lift mínimo.
                    - **Relação com CONVERSAO_LIST**: cada valor de `CONVERSAO_LIST` define um `baseline` diferente → uma linha horizontal diferente em `precision = LIFT_TARGET × conversao`. Permite visualizar a performance relativa em diferentes cenários de conversão base.
            - [P] Faria sentido avaliar a curva PR para diferentes K's e para diferentes modelos do mesmo grid? Como seria esta análise?
                - [R] Sim, faz sentido. A curva PR em si é única por modelo (independente de K). Para a comparação entre modelos:
                    - **Curvas PR por modelo**: um gráfico único com uma curva por `model_id`, cores distintas. Permite comparação visual de AUC-PR e de qual modelo domina em diferentes regimes de recall. Logado no MLflow como `comparativo/pr_curves.png`.
                    - **Pontos @K sobre as curvas**: para cada K em `K_LIST`, marcar sobre cada curva o ponto de operação (threshold de capacidade). Isso mostra onde cada modelo opera para diferentes capacidades.
                    - **Linhas de LIFT_TARGET**: linhas horizontais para cada combinação de `LIFT_TARGET × conversao` (de `CONVERSAO_LIST`), permitindo visualizar quais modelos e em qual regime de capacidade superam cada target.
                    - K's diferentes *não* geram curvas diferentes — apenas pontos diferentes sobre a mesma curva. Não faz sentido plotar uma "curva por K".
        - [P] Faz sentido gerar valores de TP@K, FP@K, TN@K e FN@K?
            - [R] Sim, com a seguinte priorização:
                - **TP@K, FP@K, FN@K**: altamente informativos. TP@K = emitidas capturadas no top-K (o que o time trabalha); FN@K = emitidas fora do top-K (oportunidades perdidas, que o modelo deveria minimizar); FP@K = perdidas no top-K (esforço desperdiçado). Devem ser incluídos.
                - **TN@K**: menos informativo para ranking — representa as perdidas que ficaram fora do top-K, o que é esperado. Pode ser incluído por completude da confusion matrix @K, mas não é o foco da análise.
                - Implementação: para cada K em `K_LIST` e cada `model_id`, calcular TP/FP/FN/TN @K usando `df_ranked.filter(col("rank") <= K)`. Logar como metrics no MLflow (ex: `tp_at_k_{pct}pct_{model_id}`) e incluir no `comparativo/metrics_summary.json`.
