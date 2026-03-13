# Direcionamento Desenvolvimento MODE_C

## Colunas COTACAO_GENERICO
['ANO_COTACAO',
 'MES_COTACAO',
 'DIA_COTACAO',
 'CD_NUMERO_COTACAO_AXA',
 'VL_PREMIO_ALVO',
 'FL_REPIQUE',
 'INTERMENDIARIO_PERFIL',
 'DT_INICIO_VIGENCIA',
 'VL_PREMIO_LIQUIDO',
 'VL_PRE_TOTAL',
 'FL_ENDOSSO',
 'FL_PROPOSTA',
 'DS_PRODUTO_NOME',
 'DS_SISTEMA',
 'CD_NUM_PROPOSTA',
 'CD_NUMERO_APOLICE_AXA',
 'CD_NUMERO_ENDOSSO_AXA',
 'DS_MOTIVO_ENDOSSO',
 'FL_ENDOSSO_RESTITUICAO',
 'VL_ENDOSSO_PREMIO_TOTAL',
 'DS_SUBSCRITOR',
 'CD_FILIAL_RESPONSAVEL_COTACAO',
 'CD_FILIAL_AXA',
 'DT_VALIDADE',
 'DS_NOME_VERSAO_CALCULO',
 'DS_STATUS',
 'DS_ATIVIDADE_SEGURADO',
 'CD_DOC_CORRETOR',
 'DS_CORRETOR_SEGMENTO',
 'DS_GRUPO_CORRETOR_SEGMENTO',
 'DS_GRUPO_STATUS',
 'FL_ANALISE_SUBSCRICAO',
 'FL_SALVO',
 'DT_ANALISE_SUBSCRICAO',
 'DT_FIM_ANALISE_SUBSCRICAO',
 'DS_REPIQUE_ACAO',
 'DS_REPIQUE_MOTIVO',
 'DS_REPIQUE_ATENDIMENTO',
 'DS_FAROL',
 'DS_TIPO_COTACAO',
 'TS_ATUALIZACAO',
 'TS_ARQ',
 'SOURCE_FILE',
 'DS_TIPO_SEGURO']

## Colunas CORRETOR_DETALHE
['CD_DOC_CORRETOR',
'DS_PRODUTO_NOME',
'DS_TIPO_SOLICITACAO',
'QTD_COTACAO_2024',
'QTD_COTACAO_2025',
'QTD_COTACAO_M2',
'QTD_COTACAO_M3',
'QTD_EMITIDO_2024',
'QTD_EMITIDO_2025',
'QTD_EMITIDO_M2',
'QTD_EMITIDO_M3',
'HR_2024',
'HR_2025',
'HR_M2',
'HR_M3',
'TS_ARQ',
'TS_ATUALIZACAO',
'SOURCE_FILE']

## Colunas CORRETOR_RESUMO
['DS_CORRETOR',
 'CD_CORRETOR',
 'DS_SEGMENTACAO',
 'DT_APROVACAO_CADASTRO',
 'DS_CANAL_COMERCIAL',
 'VL_GWP_CORRETOR',
 'FL_PERMITE_ANTECIPAR',
 'DS_SUCURSAL_AUTORIZA_EMISSAO',
 'QTD_ACORDO_COMERCIAL',
 'FL_ACORDO_GENERALISTA',
 'FL_ACORDO_ESPECIFICO',
 'DS_GRUPO_CORRETOR',
 'DS_SEGMENTACAO_GRUPO',
 'DT_ULTIMA_EMISSAO',
 'QTD_DIAS_ULTIMA_EMISSAO',
 'Qtd_PRODUTOS_HISTORICO',
 'QTD_PRODUTOS_COTADORES_ANO_ATUAL',
 'QTD_PRODUTOS_EMITIDOS_COTADORES_ANO_ATUAL',
 'HR_NOVO_2025',
 'HR_RENOVACAO_2025',
 'TS_ARQ',
 'TS_ATUALIZACAO',
 'SOURCE_FILE']








# Descrição MODE_B

> Versão de referência: V8. Input: `silver.cotacao_seg` (output do 2_JOIN). Output: tabelas gold de treino e validação + modelo MLflow.

---

## Fluxo Geral

```
silver.cotacao_seg
       │
       ▼
[PP] Pré-proc modelo      ← regras PP_Rnn / BUILD_Rnn (hoje em 1_PRE_PROC)
       │
       ├─► gold.cotacao_model      (treino,     ~80%)
       └─► gold.cotacao_validacao  (hold-out,   ~20%)
              │
              ▼
[FS] Seleção de features   ← filtra para SEG_TARGET
              │
              ▼
[TR] Treinamento           ← GBT + CV 3-fold
              │
              ▼
        Modelo MLflow + Métricas de ranking
```

---

## Etapas

### 1 — Pré-processamento do modelo (PP)
> Hoje executado em `1_PRE_PROC.py`. As regras abaixo foram movidas para upstream.

| Regra | O que faz |
|---|---|
| PP_R01 | Normaliza `DS_GRUPO_STATUS` para casing canônico |
| PP_R02 | Filtra apenas status finais (Emitida / Perdida) |
| PP_R03 | Cria coluna `label` (Emitida=1.0, Perdida=0.0) |
| BUILD_R01 | Cria coluna `MES` (yyyy-MM) a partir de DATA_COTACAO |
| BUILD_R02 | Cria flag `is_valid` via hash determinístico (ID + SEG + MES + salt) |

**Split:** 80% treino / 20% hold-out — determinístico por hash, sem aleatoriedade.

---

### 2 — Seleção de Features (FS)
> Executado sobre o conjunto de treino, filtrado para um segmento-alvo (`SEG_TARGET`).

**Candidatos configurados:**
- Numéricas: prêmios, VL_GWP_CORRETOR, dias entre datas
- Categóricas: produto, sistema, filial, atividade segurado, segmento corretor

**Pipeline de seleção:**
1. Remoção de features com >90% nulos
2. Truncagem de alta cardinalidade (>15 categorias → top 10 + OUTROS)
3. Remoção de colunas constantes
4. Encoding → Imputer (média) → VectorAssembler

**Ensemble multi-método (3 métodos × 3 seeds):**

| Método | Hiperparâmetros |
|---|---|
| Regressão Logística L1 | maxIter=100, regParam=0.01 |
| Random Forest | numTrees=200, maxDepth=8 |
| Gradient Boosting | maxIter=80, maxDepth=5, stepSize=0.1 |

- Importância por feature agregada e normalizada (rank norm)
- Peso do método = AUC-PR / soma(AUC-PR dos métodos)
- Saída: ranking final + conjuntos top_5 / top_7 / top_12
- Mutual Information calculado em paralelo (referência, não entra no ensemble)

---

### 3 — Treinamento (TR)

**Algoritmo:** Gradient Boosting Trees (GBT)

**Cross-validation:** 3 folds determinísticos via hash(ID, seed=42)

**Grid de hiperparâmetros (4 combinações):**

| maxDepth | stepSize | maxIter |
|---|---|---|
| 4 | 0.05 | 100 |
| 4 | 0.10 | 100 |
| 6 | 0.05 | 100 |
| 6 | 0.10 | 100 |

**Class weighting:** automático se `label_rate < 30%` → `weight_pos = (1 - rate) / rate`

**Seleção:** melhor combo por média de AUC-PR nos 3 folds → treino final no conjunto completo.

---

### 4 — Avaliação e Ranking

**Métricas de classificação (hold-out):**
- Average Precision (AP)
- AUC-PR (área sob a curva Precision-Recall)

**Métricas de ranking (curva de capacidade):**

| Capacidade | Métricas calculadas |
|---|---|
| 5%, 10%, 20%, K_operacional | Precision@K, Recall@K, FN@K, Lift@K |

- **Baseline de lift:** taxa de conversão do time (padrão 30%)
- **Target:** Lift ≥ 2.0× no K operacional (10% do hold-out)
- Threshold operacional derivado do K-ésimo score

---

## Estrutura MLflow

```
T_PR_TREINO  (parent container, reutilizável via override)
└── T_MODE_B  (container do mode)
    ├── T_PRE_PROC_MODEL  (container)
    │   └── T_PRE_PROC_MODEL_<TS>  ← params, métricas, regras, profiling
    ├── T_FEATURE_SELECTION  (container)
    │   └── T_FS_<TS>  ← ranking de features, importâncias por método, MI
    └── T_TREINO  (container)
        └── T_TREINO_<TS>  ← modelo, CV grid, curva de capacidade, confusion matrix
```

Tags obrigatórias nas exec runs: `pipeline_tipo`, `stage`, `run_role`, `mode`, `step`, `treino_versao`.

---

## Outputs

| Artefato | Localização |
|---|---|
| Modelo treinado | MLflow artifact `treino_final/model/` |
| Dataset treino | `gold.cotacao_model_<TS>_<UUID>` |
| Dataset hold-out | `gold.cotacao_validacao_<TS>_<UUID>` |
| Ranking de features | MLflow artifact `summary/feature_ranking_final.csv` |
| Curva de capacidade | MLflow artifact `threshold_analysis/capacity_curve.csv` |

# Implementação MODE_C


## 3_TREINO_MODE_C

<!-- > Qual a influência de seeds ao longo das etapas e das runs? Como esta questão deverá ser trabalhada e qual seu impacto no desenvolvimento e nos resultados?
    >> - **SPLIT_SALT** (string literal): determina o split treino/validação via `xxhash64(ID, SEG, MES, SPLIT_SALT)`. Não é seed numérico — é chave de particionamento. Fixo por versão do mode; muda apenas quando se quer deliberadamente outro particionamento. String literal usada como quarto input do `xxhash64` em `BUILD_R02`. Garante que splits de experimentos diferentes não se sobreponham acidentalmente mesmo com os mesmos IDs. É **por MODE (versão)**, não por execução — muda apenas quando se quer deliberadamente um particionamento diferente. Em MODE_B: `"split_b1_seg_mes"`. Em MODE_C: `"split_c1_seg_mes"`.
    >> - **FS_SEEDS** (lista, ex: `[42, 123, 7]`): usados exclusivamente no FS — cada seed gera um split 70/30 estratificado e é passado como `seed` para RF e GBT do loop de importâncias. Estabiliza o ranking de features via média entre runs. LR é determinístico e não usa o seed para treino, apenas para o split.
    >> - **CV_SEED** (int, ex: `42`): usado exclusivamente no T_TREINO — define a divisão dos folds via `xxhash64(ID, CV_SEED)` e é o `seed` do GBT final. CV_SEED=1 e CV_SEED=2 produzem partições de folds completamente diferentes. Hoje treina-se um único modelo final (não há multi-seed no treino).
    >> - **Para MODE_C**: todos os três definidos na célula Configs/Gerais, todos logados no MLflow.
    > Então, para cada treino, em um mesmo mode, eu devo manter constante o SPLIT_SALT, daí posso comparar os treinos?
        >> Sim. SPLIT_SALT fixo = mesma partição treino/validação em todas as execuções do mode → diferenças de métricas entre runs refletem apenas mudanças de features, params ou regras, não do split. Mudar o SPLIT_SALT é equivalente a mudar o "experimento base" — as runs deixam de ser comparáveis entre si. -->

<!-- - A ideia é eu poder escolher Configs de forma livre, sem comprometer as lógicas do código → manter a estrutura do MODE_B: células de Config por etapa, funções de lógica independentes de variáveis globais (recebendo parâmetros). Configs alteram os inputs das funções, não a lógica interna.
- Os logs, a princípio se mantém iguais, mas quero logar lineage da etapa de PRE_PROC_MODEL → adicionar ao log da run PRE_PROC_MODEL: `input_cotacao_seg_fqn`, `n_linhas_por_regra` (contagem após cada regra aplicada), e o catálogo de regras (já feito em MODE_B via `rules_catalog.json`). Isso constitui o lineage da etapa.
- Deixar Configs/Params/Regras por etapa, em células separadas, e, depois, ter células dedicadas à execução/logs:
    - Gerais
        - MLflow (estrutura), Versionamento, INPUT, OUTPUT
        - `SPLIT_SALT`, `CV_SEED`, `FS_SEEDS` definidos aqui
    - Pré-Processamento
        - PRE_PROC_MODEL — params (`STATUS_COL`, `LABEL_COL`, `ID_COL`, `SEG_COL`, `DATE_COL`, `VALID_FRAC`, `SPLIT_SALT`)
        - Regras de processamento definidas (Regras, Toggles, Catálogo) — mesma engine `rule_def` + `apply_rules_block`
        - Definição de FS_DECIMAL_COLS, FS_DIAS_COLS, FS_CAT_COLS: ler `df_seg.columns` logo após carregar a tabela → exibir lista → definir as três listas a partir das colunas disponíveis → usar um set `EXCLUIR_DE_FEATURES` com exclusões explícitas e documentadas. Isso evita enviar colunas que não existem e torna a escolha auditável.
        - Thresholds de drop de colunas com nulos e constantes, e de truncagem de cardinalidade, assim como 'SEG_TARGET' deverão estar definidos em Pré-Processamento.
    - Feature Selection
        - FS — params: `FS_SEEDS`, `FS_TRAIN_FRAC`, `TOPK_LIST`, listas de features a serem alimentadas ao FS
    - Treino
        - `ID_COLS` e `DROP_FROM_FEATURES` — mesma lógica do MODE_B: `ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]`; `DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]`
        - Pipeline de classificação: GBT + CV determinístico por hash — manter arquitetura do MODE_B (ver resposta detalhada abaixo) -->

---

### Etapa de Pré-Processamento

<!-- > Quais destas regras, presentes no notebook atual (3_TREINO_MODE_B) foram aplicadas em 1_PRE_PROC? Eu quero aplicá-las apenas em 3 (etapa atual)
    >> **Nenhuma** das 4 regras abaixo está em `1_PRE_PROC.py`. Todas são executadas integralmente dentro de `3_TREINO_MODE_B`, nas etapas de FS e Treino. O `1_PRE_PROC` lida com normalização de status, label e MES (PP_R01 a BUILD_R02) — upstream do modelo. As regras abaixo são específicas de preparação para ML e devem permanecer apenas em 3.
- Definir regras na etapa de PRE_PROC_MODEL do notebook 3_TREINO_MODE_C, a ser desenvolvido:
    - Remoção de features com >90% nulos — apenas em 3 (etapa FS, step 3)
    - Truncagem de alta cardinalidade (>15 categorias → top 10 + OUTROS) — apenas em 3 (etapas FS e Treino, aplicada consistentemente nas duas para evitar divergência de categorias)
    - Remoção de colunas constantes — apenas em 3 (etapa FS, step 3, via `count_distinct <= 1`)
    - Encoding → Imputer (média) → VectorAssembler — apenas em 3 (pipeline Spark ML, fit no treino, transform no treino e validação) -->

---

### Etapa de Feature Selection

> De que forma são aplicados os algoritmos de FS?
    >> Loop multi-seed × multi-método:
    >> 1. Para cada `seed` em `FS_SEEDS`: split estratificado por label (70% treino / 30% val, via `sampleBy`)
    >> 2. Pipeline de preprocessing fitado apenas no treino: StringIndexer → OHE → Imputer (média) → VectorAssembler
    >> 3. Treino e avaliação de 3 modelos: LR-L1 (coeficientes abs como importância), RF (featureImportances), GBT (featureImportances)

<!-- > Qual a lógica de fitar apenas no set de treino? De que modo estes algoritmos são utilizados para feature selection? Ou seja, se são algoritmos de classificação, por que podem ser usados como avaliadores de importância de features?

>> Fitar apenas no treino evita **data leakage**: o Imputer aprenderia a média de validação, o StringIndexer aprenderia vocabulário de categorias de validação — ambos inflam artificialmente o AUC-PR estimado. A validação deve ser tratada como dado nunca visto. Os classificadores são usados como **scorers de importância** porque a tarefa de FS é a mesma do modelo final (prever emitida/perdida). O subproduto do fit é:
>> - **RF/GBT**: importância por redução de impureza acumulada nos splits de todas as árvores onde a feature foi usada. Feature muito usada para splits que reduzem muito o erro → alta importância.
>> - **LR-L1**: coeficiente absoluto após penalização L1. L1 zera coeficientes de features irrelevantes; magnitude |coef| = força da relação linear com o label, controlando as demais features.
>> Eles classificam e, como subproduto do fit, rankeiam features por relevância para o label — daí serem úteis para FS.
    >> 4. Importâncias agregadas por feature original (somando attrs OHE), normalizadas por rank [0,1]
    >> 5. Média das importâncias entre seeds por método → ensemble ponderado por AUC-PR médio de cada método
    >> 6. Score final por feature = soma(score_método × peso_método). Saída: ranking + conjuntos top_5/7/12
    >> 7. Mutual Information calculado em paralelo (referência, não entra no ensemble)

> Existe necessidade de normalização das variáveis para os modelos de FS? Isso é feito hoje?
    >> Depende do modelo. RF e GBT são baseados em splits de árvore — invariantes à escala, não precisam de normalização. LR-L1 é sensível à escala (penalização L1 trata todas as features igualmente em magnitude). No pipeline atual, **não há `StandardScaler` explícito**, mas `LogisticRegression` é instanciado com `standardization=True` — que aplica normalização internamente, por coluna, antes de ajustar os coeficientes. O resultado prático é equivalente a StandardScaler + LR. Para MODE_C: nenhuma ação necessária — o comportamento atual está correto. -->

<!-- > Faz sentido balizar a escolha das features em quais métricas? Como é calculada a importância ensemble? O que poderia ser feito diferente?
    >> AUC-PR como peso do ensemble é coerente com o objetivo (ranking para precision-recall), pois prioriza métodos que melhor discriminam Emitida/Perdida no contexto de classes desbalanceadas. A normalização por rank (não por magnitude) evita que um método com importâncias muito grandes domine o ensemble.
    >> Pontos de melhoria possíveis:
        >> - [STDBY] Usar MI como método de desempate entre features com scores próximos (hoje é apenas referência) — viável, mas não prioritário. Implementar se o ensemble produzir empates frequentes.
        >> - Substituir split aleatório (sampleBy) por split temporal para avaliar se a feature é estável entre períodos
        >> - [DECIDIDO] Análise de correlação entre features finalistas — implementar como análise pós-FS, logada como artefato no MLflow, sem ser filtro automático. Funciona assim: após o FS selecionar o conjunto finalista (ex: top_12), calcular matrix de correlação Pearson/Spearman entre essas features; features com |r| > 0.85 são flagradas como candidatas a redundância — decisão de remoção é manual. Visualização: **heatmap de correlação** (seaborn `heatmap` ou matplotlib) salvo como `fs/correlation_matrix.png` no MLflow. -->

---

### Etapa de Treino do modelo

<!-- > Descreva o que havia sido implementado relacionado à definição de capacidade/thresholds e qual seria o sentido desta implementação (treino e resultados são afetados? qual seria o uso deste parâmetros e como eles poderiam influenciar decisões de negócio e de desenvolvimento?)
    >> `CAPACIDADE_PCT` define K = `int(n_hold_out × CAPACIDADE_PCT)` — o número de cotações que o time consegue atender. O threshold operacional é o score do K-ésimo registro no ranking do hold-out. O treino **não é afetado** por esses parâmetros — eles são usados apenas na análise pós-treino.
    >> Para decisões de negócio: permite responder "se o time atender apenas as top-K cotações sugeridas pelo modelo, qual % das conversões reais seriam capturadas?" (Recall@K) e "quantas oportunidades reais ficariam fora do radar?" (FN@K).
    >> Uso: (1) avaliar se o modelo é viável para a capacidade real do time (`Lift@K ≥ LIFT_TARGET`); (2) derivar o threshold de corte para classificação binária em produção; (3) guiar decisão de ir para produção com um dado mode.
    
    > Então o threshold operacional simplesmente retorna o score da última cotação que seria enviada ao time comercial, com base no ranking e na capacidade?

    >> Sim. Ordena o hold-out por score desc → threshold_op = score do K-ésimo registro. Toda cotação com score ≥ threshold_op seria "recomendada" ao time.

    > Como é calculado o threshold usado para definir, com base no score, se pred_emitida = 0, 1?

    >> Em MODE_B, não há threshold binário derivado separadamente — o threshold_op é usado como proxy: pred_emitida = 1 se score ≥ threshold_op. Para MODE_C, o objetivo dual sugere derivar um **threshold de classificação geral** independente, por exemplo, pelo ponto de máxima F1 na curva PR, ou pelo ponto onde Recall ≥ 80%. Este threshold de classificação seria reportado separadamente do threshold_op.

    > Existe diferença entre o threshold operacional e o threshold para pred_emitida = 0/1?

    >> Sim — são conceitualmente distintos: **threshold_op** é derivado da capacidade (K-ésimo score no ranking); **threshold_classif** é derivado de critério de performance geral (curva PR, F1, recall mínimo). Em MODE_B são usados de forma intercambiável. Para o objetivo dual de MODE_C, faz sentido reportar ambos separadamente no 5_COMP.
        > Explique detalhadamente as formas de definir o threshold de classificação

    > De que modo o LIFT_TARGET desempenha um papel nos valores calculados a partir dele? De que forma se relaciona com o threshold operacional e com as métricas de Precision/Recall@K?

    >> LIFT_TARGET é uma régua de negócio aplicada **pós-treino**, não afeta o GBT nem o grid search. Lift@K = (Recall@K) / (K/N) = taxa de acerto no top-K relativa à taxa base aleatória. LIFT_TARGET serve como critério de aceite: se Lift@K ≥ LIFT_TARGET, o modelo é considerado viável para a capacidade K do time. Relaciona-se com Precision/Recall@K indiretamente: Lift@K = Precision@K / (n_positivos/N) — maior Precision@K implica maior Lift@K. LIFT_TARGET não define o threshold_op; apenas valida se o modelo está acima da performance mínima aceitável. -->
    
<!-- - [DECIDIDO] Arquitetura do treino: GBT com CV 3-fold determinístico (hash por ID + CV_SEED) + grid search manual (4 combinações: maxDepth ∈ {4,6} × stepSize ∈ {0.05, 0.1}, maxIter fixo em 100).
    - [DECIDIDO] Mudança em relação ao MODE_B: **sem seleção automática de vencedor**. Cada combinação do grid gera um modelo salvo individualmente. O identificador de cada modelo será uma **child run MLflow aninhada na run T_TREINO**, com nome/tag `model_key = "gbt_d{maxDepth}_s{stepSize}"` (ex: `gbt_d4_s005`). O modelo é salvo como artefato da child run (`mlflow.spark.log_model`). Para referenciar depois: `run_id` da child + `model_key`.
    - [DECIDIDO] A avaliação dos modelos é feita no 5_COMP, podendo ser observados os modelos de melhor performance, após inferência nos dados de validação. O notebook 5_COMP recebe o `PR_RUN_ID` do T_TREINO, lista as child runs com `model_key`, carrega cada modelo e gera as análises.

## 4_INFERENCIA
- A inferência nos dados de validação ainda retornaria apenas uma tabela, mas com as p–emitidas e demais informações derivadas da inferência, para cada modelo da run, gerado a partir do grid. Portanto, a partir desta tabela gerada, como mencionado, na etapa 5_COMP eu quero fazer a comparação entre estes modelos da mesma run e gerar análises para entender o comportamento dos modelos e selecionar um modelo vencedor, caso eu queira usá-lo posteriormente. -->

## 5_COMP
<!-- - Definição das análises de cada modelo da run para seleção do modelo vencedor:
    - Avaliação de overfitting
        > Quais as formas de avaliar overfitting de cada modelo?

        >> Para GBT (e qualquer modelo do grid): comparar métricas **treino vs validação**: AUC-PR, AP e Lift@K. Se treino >> validação, overfitting. Abordagens:
        >> - **Gap de AUC-PR**: diferença > 0.05 entre treino e val é sinal de overfitting. Logar ambos.
        >> - **Curva PR treino vs validação sobrepostas**: gap visual entre as curvas.
        >> - **Distribuição de scores**: se a distribuição no treino for muito mais concentrada nos extremos (0 e 1) do que na validação, o modelo memorizou.
        >> Para GBT especificamente: overfitting manifesta via `maxDepth` alto ou `stepSize` alto — o grid contempla isso ao incluir `maxDepth ∈ {4, 6}` e `stepSize ∈ {0.05, 0.1}`.
        >> Logar treino/val metrics separadamente para cada modelo do grid → comparável direto no 5_COMP.
    - Métricas de performance
        - AUC-PR, Average Precision
        - Precision, Recall, Lift@K
            - Lift de cada modelo comparado ao LIFT_TARGET. Parâmetros `K_LIST` e `CONVERSAO_LIST` definidos em 5_COMP (são parâmetros de análise, não de treino).
            > Faz sentido avaliar curva de Lift para diversos K e taxas de conversão?

            >> Sim. É a **curva Lift@K** — eixo X = K (ou top-K% da base), eixo Y = Lift, uma linha por taxa de conversão base. Permite responder: "se a capacidade mudar de 50 para 80, como o lift do modelo evolui?". Logar como `comp/lift_curve.png`.

        > Faz sentido avaliar TP, FN, FP, TN @K? Estes valores dependeriam do threshold?

        >> Sim, e **não dependem do threshold de score** — dependem apenas de K (capacidade). Ordena por score desc, os top-K são tratados como pred=1:
        >> - TP@K = top-K que são realmente emitidos
        >> - FP@K = top-K que são perdidos
        >> - FN@K = emitidos fora do top-K → **o mais crítico para o negócio** (oportunidades perdidas)
        >> - TN@K = perdidos fora do top-K
        >> Recall@K = TP@K / (TP@K + FN@K); Precision@K = TP@K / K. Reportar como tabela por modelo do grid. -->
<!-- 
        - Curva Precision × Recall (geral, todos os thresholds)
            > Faria sentido gerar curva PR para diferentes atributos (e.g. DS_PRODUTO_NOME)?

            >> Sim — análise de performance por segmento (**segmented PR curves**). Identifica se o modelo performa bem para um produto mas mal para outro. Implementar como subplots ou linhas coloridas por categoria no mesmo gráfico. Útil para validação com a área de negócio. Logar como `comp/pr_curve_by_<atributo>.png`.

        - Curva Precision × Recall @K (no domínio K, não threshold de score)
            > Faz sentido ter esta curva apenas no universo dos dados que estão nos topK?

            >> A "curva PR @K" não é a curva PR clássica — é P@K e R@K como função de K variando de 1 a N. Ela **usa todos os dados** (o denominador de Recall@K inclui todos os positivos, não só os top-K). Portanto, não se restringe ao universo top-K; K é apenas o ponto de corte do ranking.

            > Faz sentido gerar para diversos modelos do grid, fixando K?

            >> Sim. Fixando K, plota pontos {P@K, R@K} para cada modelo — vê qual modelo dá o melhor tradeoff para aquela capacidade. É a comparação principal do 5_COMP.

            > Da mesma forma, faz sentido gerar a curva para diversos Ks, fixando modelo?

            >> Sim. Isso é a curva P@K vs R@K de um modelo específico — mostra como o tradeoff evolui à medida que a capacidade varia. Essencial para a decisão "vale a pena aumentar o time de X para Y cotações?".

            > Quero implementar estas visualizações por atributo (e.g. DS_PRODUTO_NOME)

            >> Viável — mesma lógica das segmented PR curves, aplicada no domínio K. Para cada categoria do atributo, calcular P@K e R@K considerando apenas os positivos daquela categoria. Logar como `comp/pr_k_by_<atributo>.png`.
        - Distribuição do score entre períodos -->