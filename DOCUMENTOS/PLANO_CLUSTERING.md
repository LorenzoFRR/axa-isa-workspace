# PLANO_CLUSTERING.md — MODE_D: PRE_PROC_MODEL + CLUSTERING

Guia de desenvolvimento iterativo do `3_TREINO_MODE_D.py`.
Documenta a lógica, decisões e considerações de cada sub-etapa.
Atualizar conforme o desenvolvimento avança.

**Status**: Em desenvolvimento
**Versão alvo**: V11.0.0

---

## Visão geral do MODE_D

MODE_D introduz `CLF_CORRETOR` como feature derivada por K-Means antes da etapa de treino.
A lógica de preprocessing é idêntica ao MODE_C. O clustering é uma etapa separada do PRE_PROC_MODEL, executada após o preprocessing e antes do split train/valid.

```
cotacao_seg (silver)
    │
    ▼
T_PRE_PROC_MODEL
    │  df_model, df_validacao — em memória (sem persistência em gold)
    │
    ▼
T_CLUSTERING_EXPLORE
    │  (sem output de tabela — apenas artifacts MLflow: elbow, silhouette)
    │
[decisão: K_FINAL = X]
    │
    ▼
T_CLUSTERING_FIT
    │  → gold.cotacao_model_d_{TS}_{UUID}      (com CLF_CORRETOR)
    │  → gold.cotacao_validacao_d_{TS}_{UUID}   (com CLF_CORRETOR)
    │
    ▼
(futuro) T_FEATURE_SELECTION → T_TREINO
```

**Apenas duas tabelas gold são criadas**, ao final do CLUSTERING_FIT. O df_model intermediário (saída do PRE_PROC_MODEL, sem CLF_CORRETOR) fica em memória Spark durante a sessão — não é persistido.

---

## Estrutura MLflow

```
T_PR_TREINO                    ← parent run (mesmo do MODE_C, ou novo)
  └─ T_MODE_D                  ← mode run — container do MODE_D
       ├─ T_PRE_PROC_MODEL      ← container
       │    └─ {TS_EXEC}        ← exec run: rules + feature prep
       ├─ T_CLUSTERING_EXPLORE  ← container
       │    └─ {TS_EXEC}        ← exec run: K_RANGE_EXPLORE → elbow + silhouette
       └─ T_CLUSTERING_FIT      ← container
            └─ {TS_EXEC}        ← exec run: K_FINAL → modelo, join, gold tables
```

PR e MODE run ficam abertos durante toda a sessão (sem `with`). Cada container e exec run usa `with mlflow.start_run(nested=True)` e fecha ao término da respectiva célula.

---

## Tabelas gold

| Tabela | Etapa que cria | Conteúdo |
|--------|---------------|----------|
| `gold.cotacao_model_d_{TS}_{UUID}` | T_CLUSTERING_FIT | df_model com CLF_CORRETOR |
| `gold.cotacao_validacao_d_{TS}_{UUID}` | T_CLUSTERING_FIT | df_validacao com CLF_CORRETOR |

Estas são as únicas tabelas persistidas no pipeline. São as entradas das etapas futuras (FS e TREINO do MODE_D).

---

## Workflow de seleção de K

O K não é decidido antes de rodar o notebook — é decidido a partir dos artifacts gerados pelo T_CLUSTERING_EXPLORE.

### Fluxo dentro de uma sessão

```
[Cell] T_PRE_PROC_MODEL
       → df_model em memória

[Cell] T_CLUSTERING_EXPLORE
       → para cada K em K_RANGE_EXPLORE: fit K-Means, computa inertia + silhouette
       → loga elbow_curve.png + silhouette_curve.png no MLflow

━━━ inspeciona MLflow → decide K_FINAL ━━━

[Cell] K_FINAL = X   # <<< AJUSTE

[Cell] T_CLUSTERING_FIT
       → fit K-Means(K_FINAL)
       → join CLF_CORRETOR em df_model + df_validacao
       → salva gold tables
```

### Re-execução de T_CLUSTERING_FIT na mesma sessão

Se após ver o heatmap e os scatters do FIT o K precisar de ajuste, basta:
1. Atualizar `K_FINAL` na célula de decisão
2. Re-rodar a célula T_CLUSTERING_FIT

df_model ainda está em memória. Uma nova exec run é aberta sob T_CLUSTERING_FIT container. As gold tables anteriores são sobrescritas (`WRITE_MODE = "overwrite"`).

---

## Etapa 1 — T_PRE_PROC_MODEL

### Lógica

Idêntica ao MODE_C. Não há mudança de regras, parâmetros ou comportamento.

Diferenças formais:
- `MODE_CODE = "D"` — aparece nas tags MLflow e no nome do MODE run
- `SPLIT_SALT` deve ser consistente entre execuções para comparabilidade com MODE_C; o default é o mesmo usado no MODE_C

### Output

df_model e df_validacao ficam em memória como variáveis Python na sessão Spark. Nenhuma tabela gold é criada nesta etapa.

### O que logar no MLflow

Idêntico ao MODE_C. Ver `3_TREINO_MODE_C.py` como referência.

---

## Etapa 2 — T_CLUSTERING_EXPLORE

### Toggle: CLUSTER_SEG_FILTER

Controla qual subconjunto de `cotacao_seg` alimenta o perfil de corretores para o K-Means.

| Valor | Comportamento | Quando usar |
|-------|--------------|-------------|
| `None` (default) | Perfil global: todos os corretores de todos os SEGs | Ponto de partida. Mais dados, perfil mais robusto |
| `"SEGURO_NOVO_MANUAL"` (ou outro SEG) | Perfil restrito ao SEG informado | Se análise mostrar que perfis de corretor diferem significativamente entre SEGs |

**Implementação:**
```python
CLUSTER_SEG_FILTER = None  # None = global | SEG_TARGET = por segmentação atual

df_perfil = spark.table(COTACAO_SEG_FQN)
if CLUSTER_SEG_FILTER is not None:
    df_perfil = df_perfil.filter(F.col(SEG_COL) == F.lit(CLUSTER_SEG_FILTER))
```

**Risco de leakage**: mínimo. `HR_2025_detalhe` e `QTD_COTACAO_2025_detalhe` são agregados históricos de 2025 (vindos de `corretor_detalhe_clean` via join no 2_JOIN). Não são outcomes das cotações do período de treino/validação.

---

### Sub-etapa 2.1 — Load e agregação do cotacao_seg

**Input**: `COTACAO_SEG_FQN` com aplicação de `CLUSTER_SEG_FILTER`.

**Por que usar cotacao_seg e não df_model?** df_model contém apenas cotações com status Emitida/Perdida do SEG em execução, excluindo corretores com cotações em status intermediários ou em outros SEGs. `cotacao_seg` garante o perfil mais completo do corretor.

**Agregação por CD_DOC_CORRETOR:**
```
GROUP BY CD_DOC_CORRETOR:
  hr_mean      = mean(HR_2025_detalhe)
  cotacao_mean = mean(QTD_COTACAO_2025_detalhe)
  n_produtos   = count_distinct(DS_PRODUTO_NOME)
```

**QTD_EMITIDO_2025_detalhe excluída**: mecanicamente derivada de HR × QTD_COTACAO.

---

### Sub-etapa 2.2 — NULL handling

**De onde vêm NULLs?** `hr_mean` e `cotacao_mean` podem ser NULL para corretores sem match em `corretor_detalhe_clean`. `n_produtos` nunca é NULL.

**NULL_STRATEGY = "drop"** (default): corretores sem histórico são excluídos do clustering. Receberão `CLF_CORRETOR = NULL` no join final — tratado pelo StringIndexer downstream com `handleInvalid="keep"`.

**NULL_STRATEGY = "impute_median"**: corretores sem histórico recebem o perfil mediano.

---

### Sub-etapa 2.3 — StandardScaler

K-Means minimiza distâncias euclidianas. Sem normalização, `cotacao_mean` (1 a centenas) dominaria as distâncias em relação a `hr_mean` (0–1) e `n_produtos` (1–N).

StandardScaler: subtrai a média e divide pelo desvio padrão de cada feature (calculado sobre o conjunto de corretores após NULL handling).

Os parâmetros do scaler (mean_ e scale_) são salvos em `clustering/scaler.pkl` no T_CLUSTERING_FIT — necessários para normalizar novos corretores na inferência.

---

### Sub-etapa 2.4 — K-Means explore (K_RANGE_EXPLORE)

Para cada K em `K_RANGE_EXPLORE` (ex: `[2, 3, 4, 5, 6, 7]`):
- Fit K-Means(K, n_init=10, random_seed=RANDOM_SEED)
- Computa inertia e silhouette score

Resultados logados como `clustering/elbow_curve.png` e `clustering/silhouette_curve.png`.

Nenhum modelo é salvo nesta etapa. Nenhuma tabela é criada.

---

### MLflow — T_CLUSTERING_EXPLORE exec run

- **Tags**: `pipeline_tipo=T`, `stage=TREINO`, `run_role=exec`, `mode=D`, `step=CLUSTERING_EXPLORE`, `treino_versao`
- **Params**: `clf_k_range_explore`, `clf_random_seed`, `clf_null_strategy`, `clf_cluster_features`, `clf_cluster_seg_filter`
- **Artifacts**: `clustering/elbow_curve.png`, `clustering/silhouette_curve.png`, `clustering/explore_results.json`

---

## Etapa 3 — T_CLUSTERING_FIT

### Sub-etapa 3.1 — Fit K_FINAL

Fit K-Means(K_FINAL, n_init=10, random_seed=RANDOM_SEED) sobre o mesmo conjunto de corretores do explore (mesma agregação, mesmo NULL handling, mesmo scaler).

**RANDOM_SEED**: garante reprodutibilidade entre execuções com os mesmos dados.

---

### Sub-etapa 3.2 — Atribuição de clusters e join

`pdf_corretor_clf` — pandas DataFrame com `CD_DOC_CORRETOR` e `CLF_CORRETOR` (string "0".."K-1").

**Conversão para string**: CLF_CORRETOR é string para ser tratado como categórico pelo StringIndexer downstream.

```python
df_model_clf     = df_model.join(cluster_map, on="CD_DOC_CORRETOR", how="left")
df_validacao_clf = df_validacao.join(cluster_map, on="CD_DOC_CORRETOR", how="left")
```

CLF_CORRETOR será NULL para corretores sem match em corretor_detalhe (NULL_STRATEGY="drop") ou com CD_DOC_CORRETOR = NULL.

**Métricas de cobertura**:
- `clf_pct_cobertura_model`: % de cotações em df_model com CLF_CORRETOR não-NULL
- `clf_pct_cobertura_valid`: % de cotações em df_validacao com CLF_CORRETOR não-NULL

Se pct_cobertura_model < 80%, revisar NULL_STRATEGY ou investigar joins.

---

### Sub-etapa 3.3 — Salvar tabelas gold

```python
df_model_clf.write.format("delta").mode("overwrite").saveAsTable(DF_MODEL_FQN)
df_validacao_clf.write.format("delta").mode("overwrite").saveAsTable(DF_VALID_FQN)
```

`WRITE_MODE = "overwrite"` permite re-rodar T_CLUSTERING_FIT com K diferente na mesma sessão sem conflito de nomes.

---

### MLflow — T_CLUSTERING_FIT exec run

- **Tags**: `pipeline_tipo=T`, `stage=TREINO`, `run_role=exec`, `mode=D`, `step=CLUSTERING_FIT`, `treino_versao`
- **Params**: `clf_k_final`, `clf_random_seed`, `clf_null_strategy`, `clf_cluster_features`, `clf_cluster_seg_filter`, `clf_df_model_fqn`, `clf_df_valid_fqn`
- **Metrics**: `clf_silhouette_score`, `clf_inertia`, `clf_n_corretores_total`, `clf_n_corretores_clustered`, `clf_n_corretores_sem_cluster`, `clf_n_cotacoes_com_clf_model`, `clf_pct_cobertura_model`, `clf_n_cotacoes_com_clf_valid`, `clf_pct_cobertura_valid`, `clf_cluster_{i}_n_corretores` para i=0..K-1
- **Artifacts**:

| Artifact | Conteúdo |
|----------|----------|
| `clustering/kmeans_model.pkl` | Modelo K-Means serializado (K_FINAL) |
| `clustering/scaler.pkl` | StandardScaler (mean_ e scale_) |
| `clustering/null_profile.json` | % NULLs por feature antes do scaling |
| `clustering/cluster_profile.json` | Centroides em escala original + n_corretores por cluster |
| `clustering/cluster_counts.png` | Qtd de corretores por cluster |
| `clustering/cluster_heatmap.png` | Centroides em escala original por feature (heatmap) |
| `clustering/scatter_hr_cotacao.png` | hr_mean vs cotacao_mean, cor = cluster |
| `clustering/scatter_hr_produtos.png` | hr_mean vs n_produtos, cor = cluster |
| `clustering/scatter_cotacao_produtos.png` | cotacao_mean vs n_produtos, cor = cluster |
| `clustering/corretor_tipico.json` | Mediana de cada feature por cluster |

---

### Análises e visualizações — referência

#### Curva elbow (`clustering/elbow_curve.png`)

Plota inertia (within-cluster sum of squares) vs K. Inertia sempre cai com mais clusters — o que importa é onde a queda desacelera abruptamente (o "cotovelo"). Delimita uma faixa razoável de K, não um valor único.

**Como ler**: o K na inflexão da curva é o candidato inicial.

#### Curva silhouette (`clustering/silhouette_curve.png`)

Plota silhouette score médio vs K. Para cada ponto: `(b - a) / max(a, b)`, onde `a` = distância média dentro do cluster, `b` = distância média ao cluster mais próximo. Score -1 a 1 — quanto mais alto, mais compactos e separados os clusters. O pico indica o K com melhor estrutura interna.

**Combinando os dois**: elbow delimita a faixa, silhouette aponta o K mais nítido. Quando convergem, a escolha é robusta.

#### Contagem por cluster (`clustering/cluster_counts.png`)

Bar chart com n_corretores por cluster. Detecta soluções degeneradas: cluster com >70% dos corretores ou <5% sinaliza K mal calibrado.

#### Heatmap de perfil (`clustering/cluster_heatmap.png`)

Cada linha = cluster, cada coluna = feature (hr_mean, cotacao_mean, n_produtos), valores = centroides em escala original. Permite nomear cada cluster: ex. "alto HR, baixo volume" = especialista de qualidade; "baixo HR, alto volume" = broker de volume.

Principal artifact de interpretação — permite atribuir rótulos de negócio antes de passar CLF_CORRETOR para o modelo.

#### Scatters (3 combinações)

Cada ponto = corretor, cor = cluster. Visualizam a geometria do clustering em projeções 2D:
- **HR vs QTD_COTACAO**: qualidade vs volume
- **HR vs n_produtos**: qualidade vs especialização
- **QTD_COTACAO vs n_produtos**: volume vs especialização

Sobreposição intensa em todos os scatters sugere ausência de estrutura clara para o K escolhido.

#### Corretor típico (`clustering/corretor_tipico.json`)

Mediana de hr_mean, cotacao_mean e n_produtos por cluster, mais n_corretores. Complementa o heatmap (centroides = médias no espaço escalado) com medianas em escala original — mais robusto a outliers.

```json
[
  { "cluster": "0", "hr_mean_median": 0.42, "cotacao_mean_median": 12.0, "n_produtos_median": 2.0, "n_corretores": 312 },
  ...
]
```

---

## Decisões em aberto

| Decisão | Status | Observação |
|---------|--------|------------|
| K_FINAL | Pendente | Decidir após inspecionar elbow + silhouette no MLflow |
| K_RANGE_EXPLORE | Default [2..7] | Ajustar se estrutura dos dados sugerir K maior |
| NULL_STRATEGY | Drop por padrão | Avaliar clf_pct_cobertura_model após 1ª execução; se <80%, considerar impute_median |
| CLUSTER_SEG_FILTER | None (global) | Testar por SEG se heatmaps mostrarem perfis muito diferentes entre SEGs |
| SPLIT_SALT | Mesmo do MODE_C | Manter para comparabilidade entre modos |

---

## Desenvolvimento iterativo

### Fase atual
- [x] PRE_PROC_MODEL implementado
- [ ] Refatorar para 3 containers MLflow separados (T_PRE_PROC_MODEL, T_CLUSTERING_EXPLORE, T_CLUSTERING_FIT)
- [ ] Remover persistência de tabelas intermediárias — apenas 2 tabelas gold ao final do FIT
- [ ] Implementar T_CLUSTERING_EXPLORE (K_RANGE_EXPLORE → elbow + silhouette)
- [ ] Implementar T_CLUSTERING_FIT (K_FINAL → modelo, join, gold tables, visualizações)

### Próximos passos (pós-validação)
- [ ] Executar → inspecionar elbow + silhouette → definir K_FINAL
- [ ] Inspecionar heatmap e scatters → nomear clusters (interpretação de negócio)
- [ ] Validar clf_pct_cobertura_model e clf_pct_cobertura_valid
- [ ] Validar label rate por cluster
- [ ] Implementar T_FEATURE_SELECTION (MODE_D) — incluindo CLF_CORRETOR em FEATURE_CANDIDATES
- [ ] Implementar T_TREINO (MODE_D)
- [ ] Comparar MODE_C vs MODE_D em 5_COMP
