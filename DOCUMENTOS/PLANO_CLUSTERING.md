# PLANO_CLUSTERING.md — MODE_D: PRE_PROC_MODEL + CLUSTERING

Guia de desenvolvimento iterativo do `3_TREINO_MODE_D.py`.
Documenta a lógica, decisões e considerações de cada sub-etapa.
Atualizar conforme o desenvolvimento avança.

**Status**: Em desenvolvimento
**Versão alvo**: V11.0.0

---

## Visão geral do MODE_D

MODE_D introduz `CLF_CORRETOR` como feature derivada por K-Means antes da etapa de treino.
A lógica de preprocessing é idêntica ao MODE_C. A diferença está na adição de uma nova etapa — **T_CLUSTERING** — entre PRE_PROC_MODEL e as etapas futuras de FS e TREINO.

```
cotacao_seg (silver)
    │
    ▼
T_PRE_PROC_MODEL
    │  → gold.cotacao_model_d_{TS}_{UUID}
    │  → gold.cotacao_validacao_d_{TS}_{UUID}
    │
    ▼
T_CLUSTERING
    │  input perfil: cotacao_seg (full, sem filtro SEG)
    │  input treino: gold.cotacao_model_d_{TS}_{UUID}
    │
    │  → gold.cotacao_model_d_clf_{TS}_{UUID}     (com CLF_CORRETOR)
    │  → gold.cotacao_validacao_d_clf_{TS}_{UUID}  (com CLF_CORRETOR)
    │
    ▼
(futuro) T_FEATURE_SELECTION → T_TREINO
```

---

## Estrutura MLflow

```
T_PR_TREINO                  ← parent run (mesmo do MODE_C, ou novo)
  └─ T_MODE_D                ← mode run — container do MODE_D
       ├─ T_PRE_PROC_MODEL   ← container da etapa
       │    └─ {TS_EXEC}     ← exec run — onde os logs ficam
       └─ T_CLUSTERING       ← container da etapa
            └─ {TS_EXEC}     ← exec run — onde os logs ficam
```

Todas as etapas são nested dentro do T_MODE_D. O PR e o MODE run ficam abertos durante toda a execução do notebook (sem `with`), e cada step usa `with mlflow.start_run(nested=True)`.

---

## Etapa 1 — T_PRE_PROC_MODEL

### Lógica

Idêntica ao MODE_C. Não há mudança de regras, parâmetros ou comportamento.

Diferenças formais:
- `MODE_CODE = "D"` — aparece nas tags MLflow e no nome do MODE run
- Tabelas de saída incluem `_d_` no nome para distinguir do MODE_C
- `SPLIT_SALT` deve ser consistente entre execuções para comparabilidade com MODE_C; o default é o mesmo usado no MODE_C

### Output

| Tabela | FQN |
|--------|-----|
| df_model | `gold.cotacao_model_d_{TS}_{UUID}` |
| df_validacao | `gold.cotacao_validacao_d_{TS}_{UUID}` |

### Override de execução

`DF_MODEL_FQN_OVERRIDE` + `DF_VALID_FQN_OVERRIDE`: se ambos preenchidos, o notebook pula a etapa PRE_PROC_MODEL e usa as tabelas indicadas diretamente no T_CLUSTERING. Útil para iterar no clustering sem reprocessar o pipeline inteiro.

Quando o override é ativo, `DF_MODEL_FQN` e `DF_VALID_FQN` assumem os valores do override.

### O que logar no MLflow

Idêntico ao MODE_C. Ver `3_TREINO_MODE_C.py` como referência.

---

## Etapa 2 — T_CLUSTERING

### Sub-etapa 2.1 — Load do cotacao_seg para perfil de corretores

**Input**: `COTACAO_SEG_FQN` (full, sem filtro SEG por padrão)

**Por que usar cotacao_seg completo e não df_model?**

df_model contém apenas o split de treino (is_valid=False) e apenas cotações com status Emitida/Perdida. Isso pode excluir:
- Corretores que aparecem apenas na validação
- Corretores com cotações em status intermediários

Usar cotacao_seg completo garante o perfil mais robusto e completo de cada corretor.

**Risco de leakage**: mínimo. `HR_2025_detalhe` e `QTD_COTACAO_2025_detalhe` são agregados históricos de 2025 (vindos de `corretor_detalhe_clean` via join no 2_JOIN). Não são outcomes das cotações do período de treino.

`CLUSTER_SEG_FILTER` (default: None): permite restringir o perfil a um SEG específico caso se observe que os perfis de corretor diferem muito entre SEGs. Testar primeiro com None.

**Colunas selecionadas para agregação**: `CD_DOC_CORRETOR`, `DS_PRODUTO_NOME`, `HR_2025_detalhe`, `QTD_COTACAO_2025_detalhe`

---

### Sub-etapa 2.2 — Agregação por CD_DOC_CORRETOR

```
GROUP BY CD_DOC_CORRETOR:
  hr_mean      = mean(HR_2025_detalhe)
  cotacao_mean = mean(QTD_COTACAO_2025_detalhe)
  n_produtos   = count_distinct(DS_PRODUTO_NOME)
```

**hr_mean**: média do HR do corretor through de todos os seus produtos. HR_2025_detalhe é computado no nível (corretor, produto, tipo_solicitacao), então hr_mean captura a qualidade de conversão média cross-produto.

**cotacao_mean**: volume médio de cotações por produto. Não é a soma total — a soma seria distorcida para corretores com muitos produtos.

**n_produtos**: conta produtos distintos em que o corretor aparece na base. É a operacionalização do eixo "generalista/especialista".

**QTD_EMITIDO_2025_detalhe excluída**: redundante, pois é mecanicamente derivada de HR × QTD_COTACAO.

**Logar**: total de corretores únicos, % NULLs por feature.

---

### Sub-etapa 2.3 — NULL handling

**De onde vêm NULLs?** `HR_2025_detalhe` e `cotacao_mean` podem ser NULL para corretores sem match na tabela `corretor_detalhe_clean` (join left no 2_JOIN não encontrou o corretor). `n_produtos` nunca é NULL (count distinct retorna 0 ou mais).

**NULL_STRATEGY = "drop"** (default): corretores sem histórico _detalhe são excluídos do clustering. Eles receberão `CLF_CORRETOR = NULL` após o join com df_model.

**NULL_STRATEGY = "impute_median"**: corretores sem histórico recebem o perfil mediano. Agrupa-os no cluster "médio" — pode ser aceitável ou não dependendo do contexto.

**Impacto no join final**: CLF_CORRETOR será NULL para:
- Corretores sem match em corretor_detalhe
- Cotações com CD_DOC_CORRETOR = NULL (intermediários sem documento)

Esse NULL é tratado pelo StringIndexer do pipeline downstream com `handleInvalid="keep"` (categoria extra para valores desconhecidos).

**Decisão em aberto**: avaliar % de cotações sem CLF_CORRETOR após o join. Se for alta (>20%), considerar impute_median.

---

### Sub-etapa 2.4 — StandardScaler

K-Means minimiza distâncias euclidianas. Sem normalização, features com maior variância (cotacao_mean pode ser de 1 a centenas) dominam as distâncias, enquanto hr_mean (0-1) e n_produtos (1-N) ficam com peso desprezível.

**StandardScaler**: para cada feature, subtrai a média e divide pelo desvio padrão do conjunto de corretores (após NULL handling).

**Salvar o scaler como artifact**: os parâmetros de scaling (mean_ e scale_) devem ser salvos em `clustering/scaler.pkl` para uso na inferência — novos corretores precisam ser normalizados com os mesmos parâmetros do treino.

---

### Sub-etapa 2.5 — K-Means fit

**K_FINAL**: definido com base na análise exploratória em `TESTE_CLUST_1.py`. Default = 4.

**n_init=10**: testa 10 inicializações aleatórias diferentes e mantém a melhor (menor inertia). Reduz a sensibilidade ao ponto de partida do K-Means.

**RANDOM_SEED**: garante reprodutibilidade entre execuções com os mesmos dados.

**Salvar o modelo como artifact**: `clustering/kmeans_model.pkl`. Na inferência, o centroide de cada cluster deve estar disponível para atribuir novos corretores.

**Logar no MLflow**: silhouette_score, inertia, n_corretores_clustered, distribuição de corretores por cluster.

---

### Sub-etapa 2.6 — Atribuição de clusters e join

**Resultado**: `pdf_corretor_clf` — pandas DataFrame com `CD_DOC_CORRETOR` e `CLF_CORRETOR` (int 0..K-1).

**Conversão para string**: CLF_CORRETOR deve ser convertido para string antes de entrar no pipeline de treino. O pipeline usa StringIndexer para categóricas — int seria tratado como numérico.

```
df_model_clf    = df_model.join(cluster_map, on="CD_DOC_CORRETOR", how="left")
df_validacao_clf = df_validacao.join(cluster_map, on="CD_DOC_CORRETOR", how="left")
```

**Métricas importantes após o join**:
- `n_cotacoes_com_clf`: cotações que receberam cluster
- `n_cotacoes_sem_clf`: cotações com CLF_CORRETOR = NULL
- `pct_cobertura_model`: cobertura em df_model
- `pct_cobertura_valid`: cobertura em df_validacao

Se pct_cobertura_model for baixo (<80%), revisar NULL_STRATEGY ou investigar joins.

---

### Sub-etapa 2.7 — Salvar tabelas enriquecidas

| Tabela | FQN | Conteúdo |
|--------|-----|----------|
| df_model_clf | `gold.cotacao_model_d_clf_{TS}_{UUID}` | df_model + CLF_CORRETOR |
| df_validacao_clf | `gold.cotacao_validacao_d_clf_{TS}_{UUID}` | df_validacao + CLF_CORRETOR |

Estas são as tabelas de entrada das etapas futuras (FS e TREINO do MODE_D).

---

### MLflow — o que logar no T_CLUSTERING exec run

**Tags**: `pipeline_tipo=T`, `stage=TREINO`, `run_role=exec`, `mode=D`, `step=CLUSTERING`, `treino_versao`

**Params**:
- `k_final`, `random_seed`, `null_strategy`
- `cluster_features` (lista das 3 features)
- `cluster_seg_filter`
- `cotacao_seg_fqn`, `df_model_fqn`, `df_valid_fqn`
- `df_model_clf_fqn`, `df_valid_clf_fqn`

**Metrics**:
- `silhouette_score`, `inertia`
- `n_corretores_total`, `n_corretores_clustered`, `n_corretores_sem_cluster`
- `n_cotacoes_model`, `n_cotacoes_com_clf_model`, `pct_cobertura_model`
- `n_cotacoes_valid`, `n_cotacoes_com_clf_valid`, `pct_cobertura_valid`
- `cluster_{i}_n_corretores` para i=0..K-1

**Artifacts**:
- `clustering/kmeans_model.pkl` — modelo K-Means serializado
- `clustering/scaler.pkl` — StandardScaler serializado
- `clustering/cluster_profile.json` — centroides em escala original + contagem por cluster
- `clustering/null_profile.json` — NULLs por feature antes do scaling
- `tables_lineage.json`

---

## Decisões em aberto

| Decisão | Status | Observação |
|---------|--------|------------|
| K_FINAL | Pendente | Definir após análise em TESTE_CLUST_1.py |
| NULL_STRATEGY | Drop por padrão | Avaliar pct_cobertura_model após primeira execução |
| CLUSTER_SEG_FILTER | None (global) | Testar por SEG se perfis diferirem muito |
| SPLIT_SALT | Mesmo do MODE_C | Manter para comparabilidade entre modos |

---

## Desenvolvimento iterativo

### Fase atual
- [x] PRE_PROC_MODEL implementado
- [x] T_CLUSTERING implementado (básico)

### Próximos passos (pós-validação)
- [ ] Executar e validar pct_cobertura
- [ ] Validar label rate por cluster (INTERPRETACAO_CLUSTERING.md — seção 4)
- [ ] Implementar T_FEATURE_SELECTION (MODE_D) — incluindo CLF_CORRETOR em FEATURE_CANDIDATES
- [ ] Implementar T_TREINO (MODE_D)
- [ ] Comparar MODE_C vs MODE_D em 5_COMP
