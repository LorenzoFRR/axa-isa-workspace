# ANALISE_V11 — CLF_CORRETOR (Classificação Perfil Corretor)

Documento iterativo de análise exploratória para criação da feature `CLF_CORRETOR`.
Atualizado conforme avanço da análise.

**Status**: Em andamento
**Versão pipeline alvo**: V11.0.0
**Input**: `silver.cotacao_seg_{TS}` (mesma tabela usada pelo 3_TREINO_MODE_C)

---

## 1. Objetivo

Criar uma feature categórica composta (`CLF_CORRETOR`) que classifica o perfil do corretor a partir de múltiplos eixos, capturando interações entre features que hoje entram no modelo separadamente.

CLF_CORRETOR **não substitui** as features individuais — será uma feature adicional, controlada pelo mesmo sistema de toggles (`FEATURE_CANDIDATES`).

---

## 2. Eixos / Abordagem

### Decisão: K-Means sobre vetor por corretor

A abordagem de thresholds manuais por eixo (alto/baixo) foi descartada por ser subjetiva. A abordagem adotada é **K-Means** sobre um vetor de features agregadas por corretor.

**Features de clustering** (agregação de `cotacao_seg` por `CD_DOC_CORRETOR`):

| Feature | Agregação | Captura |
|---------|-----------|---------|
| `HR_2025_detalhe` | mean | qualidade de conversão |
| `QTD_COTACAO_2025_detalhe` | mean | volume de atividade |
| `n_produtos_distintos` | count distinct `DS_PRODUTO_NOME` | breadth (generalista/especialista) |

**`QTD_EMITIDO_2025_detalhe` descartada** — é mecanicamente derivada de HR × QTD_COTACAO, introduziria redundância estrutural no vetor de clustering.

**Fluxo**: `cotacao_seg` → agregar por `CD_DOC_CORRETOR` → StandardScaler → K-Means → `CLF_CORRETOR` (int) por corretor → join de volta nas cotações.

**Input**: `cotacao_seg` (não `corretor_detalhe_clean`) — já contém todas as features necessárias e é consistente com o fluxo de join-back ao pipeline.

---

## 3. Justificativa ML

- **Correlação entre features**: HR = QTD_EMITIDO / QTD_COTACAO → as 3 features _detalhe são matematicamente correlacionadas. Combinar em feature composta pode reduzir redundância
- **Interações pré-codificadas**: o GBT (maxDepth=4-6) tem budget limitado de profundidade. Pré-codificar interações é mais eficiente
- **Informação nova**: a dimensão generalista/especialista não existe hoje como feature no pipeline
- **Perda de informação**: risco de descartar variância intra-bin ao binarizar contínuas. Mitigado por manter features individuais disponíveis

---

## 4. Plano de Análise

### Passo 1 — Executar TESTE_CLUST_V11.py
- [ ] Configurar `INPUT_TABLE` e `SEG_FILTER` na célula Config
- [ ] Verificar NULLs no profiling (célula de agregação)
- [ ] Avaliar elbow curve + silhouette para K=3,4,5
- [ ] Escolher K_FINAL e re-executar fit final
- [ ] Inspecionar heatmap de centroides e scatter plots
- [ ] Nomear clusters semanticamente com base nos centroides

### Passo 2 — Validar discriminação
- [ ] Para cada cluster, calcular label rate (taxa de Emitida) após join com cotações
- [ ] Confirmar que os clusters discriminam a variável-alvo

### Passo 3 — Mutual Information
- [ ] MI de CLF_CORRETOR vs label
- [ ] Comparar com MI de HR_2025_detalhe e QTD_COTACAO_2025_detalhe individualmente

---

## 5. Resultados

*(Serão preenchidos conforme a análise avança)*

### 5.1 Distribuições
_(pendente)_

### 5.2 Correlações
_(pendente)_

### 5.3 Thresholds definidos
_(pendente)_

### 5.4 Label rate por eixo
_(pendente)_

### 5.5 Cardinalidade
_(pendente)_

### 5.6 Mutual Information
_(pendente)_

---

## 6. Decisões

| Decisão | Status | Resultado |
|---------|--------|-----------|
| Abordagem (manual vs clustering) | ✅ Definida | K-Means — elimina thresholds subjetivos |
| QTD_EMITIDO_2025_detalhe | ✅ Descartada | Redundante (HR × QTD_COTACAO) |
| Features de clustering | ✅ Definidas | hr_mean, cotacao_mean, n_produtos_distintos |
| Input table para clustering | ✅ Definida | cotacao_seg (não corretor_detalhe_clean) |
| K (número de clusters) | Pendente | Avaliar via TESTE_CLUST_V11.py |
| Merge de clusters raros | Pendente | — |
| CLF_CORRETOR viável para pipeline | Pendente | — |

---

## 7. Próximos Passos (pós-análise)

Se viável:
- Integrar como regra PP_R08 no `3_TREINO_MODE_C.py`
- Adicionar ao `FEATURE_CANDIDATES` com toggle
- Persistir centroides K-Means como artifact MLflow (para reproducibilidade na inferência)
- Atualizar inferência (`4_INFERENCIA_MODE_C.py`) — carregar centroides e atribuir cluster a novos corretores
