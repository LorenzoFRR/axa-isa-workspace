# Interpretação de Resultados — Clustering de Corretores (CLF_CORRETOR)

Guia prático para interpretar os outputs do `TESTE_CLUST_1.py`.
Escrito para quem nunca trabalhou com clustering.

---

## 1. O que K-Means faz (em uma frase)

K-Means agrupa corretores de forma que, dentro de cada grupo, os corretores sejam o mais parecidos possível entre si — e o mais diferentes possível dos outros grupos — usando apenas as 3 features de perfil (HR, volume de cotações, diversidade de produtos). O label (Emitida/Perdida) **não entra** no clustering.

---

## 2. O conceito de "centroide"

Cada cluster tem um **centroide**: o ponto médio de todos os corretores que pertencem a ele. É o "corretor típico" daquele grupo. Por exemplo:

```
Centroide do Cluster 2:
  hr_mean      = 0.38   → taxa de conversão de 38%
  cotacao_mean = 120.5  → em média 120 cotações por produto
  n_produtos   = 1.2    → atua em praticamente 1 produto só
```

Você interpreta o cluster pelo perfil do seu centroide.

---

## 3. Outputs do notebook — como ler cada um

### 3.1 Elbow curve (curva de cotovelo)

**O que é**: gráfico de inertia por K.
**Inertia** = soma das distâncias de cada corretor ao centroide do seu cluster. Quanto menor, mais "compactos" são os clusters.

**Como ler**: à medida que K aumenta, a inertia sempre cai (mais grupos = clusters menores). O ponto de interesse é onde a queda começa a ser pequena — o "cotovelo" da curva. Adicionar mais clusters além desse ponto não traz ganho proporcional.

```
Exemplo de leitura:
  K=3 → inertia 850
  K=4 → inertia 620  ← queda grande, vale o cluster extra
  K=5 → inertia 590  ← queda pequena, K=4 provavelmente suficiente
```

**Decisão**: se a curva tiver um cotovelo claro em K=4, use K=4.
Se for suave (sem cotovelo claro), olhe o silhouette para desempatar.

---

### 3.2 Silhouette Score

**O que é**: mede quão bem cada corretor se encaixa no seu próprio cluster em comparação com o cluster mais próximo.

**Escala**: de -1 a +1.
- `> 0.5` → clusters bem separados e coesos
- `0.2 a 0.5` → estrutura razoável, aceitável para dados reais
- `< 0.2` → clusters muito sobrepostos, K pode estar errado
- `negativo` → pontos mal classificados (sinal de problema)

**Como usar em conjunto com o elbow**: o K ideal é aquele com o **menor salto de melhoria na inertia** que ainda **mantém silhouette alto**. Geralmente esses dois critérios apontam para o mesmo K.

---

### 3.3 Heatmap de centroides

**O que é**: tabela visual onde cada linha é um cluster e cada coluna é uma feature. A cor representa quão alto ou baixo é o valor daquela feature naquele cluster em relação à média global.

```
Verde escuro = muito acima da média global
Vermelho escuro = muito abaixo da média global
Amarelo/branco = próximo à média
```

Os **valores impressos nas células** são na escala original (antes da normalização) — use-os para interpretar com números reais.

**Como nomear os clusters a partir do heatmap**:

Exemplo hipotético com K=4:

| Cluster | hr_mean | cotacao_mean | n_produtos | Interpretação sugerida |
|---------|---------|--------------|------------|------------------------|
| 0 | 🟢 alto | 🟢 alto | 🔴 baixo | Especialista eficiente — alto volume, alta conversão, 1 produto |
| 1 | 🔴 baixo | 🔴 baixo | 🟡 médio | Corretor periférico — pouco ativo, baixa conversão |
| 2 | 🟢 alto | 🔴 baixo | 🟢 alto | Generalista eficiente — atua em muitos produtos, converte bem |
| 3 | 🔴 baixo | 🟢 alto | 🟡 médio | Volume alto, baixa qualidade — envia muito, converte pouco |

**Os rótulos são seus** — o algoritmo devolve apenas números (0, 1, 2, 3). A interpretação semântica vem do heatmap.

---

### 3.4 Scatter plots

Dois gráficos: HR vs Volume de cotações, e HR vs Diversidade de produtos. Cada ponto é um corretor, colorido pelo cluster.

**O que procurar**: clusters visualmente separados no espaço. Se os pontos de cores diferentes estiverem muito misturados, os clusters não têm boa separação geométrica — sinal para testar K diferente.

**É normal** haver alguma sobreposição nos bordos entre clusters vizinhos — K-Means força fronteiras lineares.

---

### 3.5 Tabela de corretores representativos

Lista os 3 corretores mais próximos ao centroide de cada cluster. São os "exemplares" mais típicos do grupo.

**Uso prático**: pegue esses IDs e consulte o histórico real desses corretores para validar se o perfil faz sentido de negócio. Se o "corretor especialista eficiente" for de fato um corretor que você conhece como de alto desempenho, o cluster está capturando algo real.

---

## 4. A validação mais importante: label rate por cluster

Depois de ter os clusters, a pergunta central para o projeto ISA é:

> **Corretores de clusters diferentes têm taxas de conversão diferentes?**

Se sim, `CLF_CORRETOR` tem poder discriminativo e vale como feature.

Como calcular (usando a célula opcional do notebook):

```python
df_resultado_sp = spark.createDataFrame(df_resultado)
df_com_clf = spark.table(INPUT_TABLE).join(df_resultado_sp, on="CD_DOC_CORRETOR", how="left")

display(
    df_com_clf
    .groupBy("CLF_CORRETOR", "DS_GRUPO_STATUS")
    .count()
    .orderBy("CLF_CORRETOR")
)
```

O que você quer ver:
```
Cluster 0 → 45% Emitida  ← conversor forte
Cluster 1 → 12% Emitida  ← conversor fraco
Cluster 2 → 31% Emitida
Cluster 3 → 22% Emitida
```

Se as taxas forem próximas entre todos os clusters (ex: 22%, 24%, 23%, 25%), o clustering não captura variação relevante para o label — e CLF_CORRETOR não vai ajudar o modelo.

---

## 5. O que avaliar antes de decidir usar CLF_CORRETOR no pipeline

| Check | O que olhar | Sinal positivo |
|-------|------------|----------------|
| Silhouette | ≥ 0.25 | Clusters coesos o suficiente |
| Elbow | Cotovelo visível | K escolhido tem suporte estatístico |
| Heatmap | Clusters com perfis distintos entre si | Cada cluster tem "personalidade" própria |
| Tamanho dos clusters | Nenhum cluster com < 20 corretores | Evita categorias raras |
| Label rate por cluster | Diferença ≥ 10pp entre clusters | CLF_CORRETOR discrimina o target |
| Sanity check | Corretores representativos fazem sentido | Valida com conhecimento de negócio |

---

## 6. Limitações a ter em mente

**Os labels são arbitrários**: "Cluster 0" não significa nada sozinho. O significado vem do heatmap. Se você re-executar com uma seed diferente, os números podem trocar (Cluster 0 vira Cluster 2). O perfil permanece, o número muda.

**K-Means assume clusters esféricos**: se o espaço de corretores tiver grupos com formas irregulares, K-Means pode não capturá-los bem. Os scatter plots ajudam a detectar isso.

**Corretores sem histórico _detalhe**: com `NULL_STRATEGY="drop"`, esses corretores ficam sem CLF_CORRETOR no join final. Isso precisa ser tratado no pipeline (provavelmente categoria "sem_historico").

**Clustering é feito no período histórico**: os centroides são aprendidos nos dados de treino. Na inferência, novos corretores são atribuídos ao centroide mais próximo — não é re-treinado. Isso é correto e desejado.

---

## 7. Revalidação do código — pontos de atenção

O código está correto e bem estruturado. Dois pontos menores a corrigir:

1. **`from scipy.spatial.distance import cdist`** está dentro da célula de análise (linha 254), não na célula de imports. Se `scipy` não estiver no ambiente, o erro vai aparecer tarde. Recomendo mover para a célula de imports.

2. **`corretor_ids`** (linha 134) é atribuído mas nunca usado — os IDs vêm diretamente de `pdf["CD_DOC_CORRETOR"]` no restante do código. Pode ser removido.

Nenhum dos dois afeta o resultado.
