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

## 2. Eixos Propostos

4 eixos, cada um parametrizável (ativável/desativável):

| Eixo | Feature base | Classificação | Descrição |
|------|-------------|---------------|-----------|
| HR | `HR_2025_detalhe` | alto_HR / baixo_HR | Hit rate do corretor para o produto |
| EMITIDO | `QTD_EMITIDO_2025_detalhe` | alto_EMIT / baixo_EMIT | Volume de apólices emitidas |
| COTACAO | `QTD_COTACAO_2025_detalhe` | alto_COT / baixo_COT | Volume total de cotações |
| GENERALISTA | count distinct `DS_PRODUTO_NOME` por `CD_DOC_CORRETOR` | generalista / especialista | Diversidade de produtos do corretor |

**Nota**: as colunas `_detalhe` vêm do join no nível produto (CD_DOC_CORRETOR + DS_PRODUTO_NOME + DS_TIPO_COTACAO). A dimensão GENERALISTA requer contagem cross-produto por corretor.

---

## 3. Justificativa ML

- **Correlação entre features**: HR = QTD_EMITIDO / QTD_COTACAO → as 3 features _detalhe são matematicamente correlacionadas. Combinar em feature composta pode reduzir redundância
- **Interações pré-codificadas**: o GBT (maxDepth=4-6) tem budget limitado de profundidade. Pré-codificar interações é mais eficiente
- **Informação nova**: a dimensão generalista/especialista não existe hoje como feature no pipeline
- **Perda de informação**: risco de descartar variância intra-bin ao binarizar contínuas. Mitigado por manter features individuais disponíveis

---

## 4. Plano de Análise

### Passo 1 — Distribuições univariadas
- [ ] Histogramas + percentis (p10, p25, p50, p75, p90) para HR, QTD_EMITIDO, QTD_COTACAO
- [ ] Distribuição de n_produtos_distintos por CD_DOC_CORRETOR
- [ ] Segmentar por SEG (SEGURO_NOVO_MANUAL, RENOVACAO_MANUAL)
- [ ] Documentar NULLs e edge cases (HR=0, HR=1)

### Passo 2 — Correlação entre features _detalhe
- [ ] Pearson + Spearman: QTD_COTACAO ↔ QTD_EMITIDO ↔ HR
- [ ] Confirmar redundância esperada

### Passo 3 — Definição de thresholds
- [ ] Propor thresholds por mediana (ponto de partida)
- [ ] Comparar com p25, p75
- [ ] Definir N para generalista (e.g., 3+ produtos)

### Passo 4 — Label rate por eixo
- [ ] Para cada eixo (alto/baixo), calcular contagem e taxa de conversão
- [ ] Validar discriminação da variável-alvo

### Passo 5 — Cardinalidade de CLF_CORRETOR
- [ ] Computar CLF_CORRETOR com 4 eixos
- [ ] Contar ocorrências por combinação
- [ ] Identificar combinações raras (< 50 cotações)
- [ ] Testar variantes com 3 eixos (removendo um por vez)

### Passo 6 — Mutual Information
- [ ] MI de cada feature individual vs label
- [ ] MI de CLF_CORRETOR (diversas combinações de eixos) vs label
- [ ] Comparar ganho informacional

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
| Quantos eixos usar (3 vs 4) | Pendente | — |
| Thresholds por eixo | Pendente | — |
| Merge de categorias raras | Pendente | — |
| CLF_CORRETOR viável para pipeline | Pendente | — |

---

## 7. Próximos Passos (pós-análise)

Se viável:
- Integrar como regra PP_R08 no `3_TREINO_MODE_C.py`
- Adicionar ao `FEATURE_CANDIDATES` com toggle
- Persistir thresholds como artifact MLflow
- Atualizar inferência (`4_INFERENCIA_MODE_C.py`)
