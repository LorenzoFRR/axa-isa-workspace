# Resumo Executivo — Intelligent Score Agent (ISA) | AXA

---

## 1) Desafio e Solução

### Desafio
- O time comercial opera com capacidade limitada: não é possível trabalhar todas as cotações com a mesma intensidade.
- Sem priorização estruturada, o esforço pode ser distribuído de forma subótima — atendendo cotações com baixa chance de conversão enquanto oportunidades melhores ficam em segundo plano.

### Solução
- O ISA classifica as cotações por probabilidade de conversão (emissão de apólice), gerando um ranking que orienta a atuação do time comercial.
- O sistema é desenvolvido de forma iterativa: cada versão é treinada, avaliada e comparada com versões anteriores antes de qualquer mudança operacional. Isso garante que evoluções sejam embasadas em dados.
- A estrutura é flexível: novos modelos, segmentos ou critérios de avaliação podem ser incorporados sem redesenho da solução.

### Como funciona para o time comercial
O output do sistema é uma **lista de cotações ordenada por score** — da maior para a menor probabilidade de conversão. O time trabalha de cima para baixo, priorizando automaticamente as oportunidades mais promissoras dentro da capacidade do dia.

O sistema não substitui o julgamento comercial — ele estrutura a fila de trabalho com base em evidências históricas.

---

## 2) Metodologia (em linguagem de negócio)

O sistema aprende a partir do **histórico de cotações com desfecho conhecido** — cotações que já resultaram em emissão de apólice ou em perda. Com esses dados, ele identifica padrões que diferenciam os dois grupos.

A avaliação leva em conta dois objetivos simultâneos:
- **Eficácia dentro da capacidade do time**: das N cotações que o time consegue trabalhar por dia, quantas são realmente convertíveis? O sistema tenta maximizar esse aproveitamento.
- **Qualidade geral do modelo**: o sistema também deve distinguir bem emissões de perdas no universo completo, não apenas no topo da lista.

Para garantir transparência e comparabilidade, múltiplas configurações de modelo são testadas lado a lado. Cada execução é rastreada com métricas, artefatos e parâmetros registrados — permitindo que qualquer resultado seja reproduzido ou auditado.

### Segmentações
O modelo é desenvolvido separadamente por segmento de negócio, respeitando as diferenças de comportamento entre eles:

| Segmento | Status |
|---|---|
| Seguro Novo Manual | Treinado e avaliado (em andamento — versões V9.0.0 e V10.0.0) |
| Seguro Novo Digital | Pendente |
| Renovação Manual | Pendente |
| Renovação Digital | Pendente |

A estrutura já construída permite adicionar os demais segmentos sem retrabalho arquitetural. É possível ainda estratificar por produto dentro de cada segmento, acomodado na mesma estrutura.

---

## 3) Estado atual

- **Pipeline completo de treinamento operacional**: ingestão de dados → aplicação de regras de negócio → treinamento de modelos → scoring → comparação de versões. Cada etapa é rastreada e documentada automaticamente, com acesso disponível via Databricks.
- **Segmento Seguro Novo Manual**: múltiplas versões de modelo treinadas e avaliadas (V9.0.0, V10.0.0), com resultados comparativos disponíveis para análise — incluindo performance ajustada para rankeamento (top-K) e classificação geral.
- **Rastreabilidade total**: cada execução registra quais dados foram usados, quais regras foram aplicadas, quais métricas foram obtidas e quais modelos foram gerados — garantindo reprodutibilidade e auditabilidade.
- **Pipeline de produção (inferência sobre cotações sem desfecho)**: pendente de implementação. É a etapa que viabilizará o uso operacional do sistema pelo time comercial.

A partir de interações com a área executiva e o time comercial, escopo, critérios de avaliação e segmentações podem ser ajustados iterativamente.

---

## 4) Próximos passos

| Ação | Objetivo |
|---|---|
| Análise de capacidade operacional do time | Calibrar o sistema com o volume real de cotações trabalhadas por período — esse dado é central para a avaliação de performance |
| Treinar e avaliar os demais segmentos | Ampliar cobertura para o universo completo de cotações |
| Implementar pipeline de produção | Habilitar o uso operacional: score de cotações sem desfecho conhecido e distribuição para o time |
| Iterações com área de negócio | Incorporar feedbacks dos usuários para direcionar próximas versões |

### O que é necessário do lado AXA
- **Capacidade operacional do time**: quantas cotações por segmento o time consegue trabalhar por período (dia/semana). Esse número calibra diretamente a avaliação do modelo — sem ele, não é possível medir se o sistema está gerando valor real.
- **Acesso a cotações em aberto**: dados de cotações sem desfecho conhecido, necessários para o pipeline de produção (inferência operacional).
