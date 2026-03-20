> Inserir elemento 💡
> Desenvolvido ✅

# 1 - Desafio e Solução - Abordagem que tem sido tomada pelo time de desenvolvimento da PSW

## Desafio interpretado
- Necessidade de abordagem data-driven para priorização de cotações de maior potencial de aprovação
- Limitação de operação do time comercial

## Solução (já implementada, à espera de testes e feedbacks da área comercial e usuários)
- Sistema transparente, observável e escalável de rankeamento de cotações implementado no ambiente Databricks

💡 Inserir diagrama de fluxo simplificado
- Pipeline de treinamento ✅
- Pipeline de produção ✅
- Apontar importância estrutura MLflow para sustentação do desenvolvimento
    - Inclusive acomoda mudanças de direcionamento a partir de decisões executivas
- Apontar outputs do time comercial como força motriz do direcionamento, além do desenvolvimento interno constante do ciclo de vida dos modelos e análises de resultados
    - Ciclo de evolução — desenvolvimento → produção → uso real pelo time → feedback e novos dados → próxima iteração.
    - Os insights do time que usa a solução, somados às análises de performance do sistema, informam diretamente a próxima versão do modelo — tornando-o cada vez mais aderente à operação da AXA.
- Distribuição de tabela com cotações rankeadas via dashboard

## Segmentações
- Hoje, existem 4 segmentações
    - SEGURO_NOVO_DIGITAL
    - SEGURO_NOVO_MANUAL
    - RENOVACAO_DIGITAL
    - RENOVACAO_MANUAL
- Idealmente, é desenvolvido um modelo por segmentações, podendo ser criados mais modelos (estratificados por produtos, por exemplo), acomodados na estrutura já criada, que confere velocidade de desenvolvimento e observabilidade para AXA. 

## Pipeline desenvolvimento

💡 Inserir fluxograma com as etapas + paralelo com estrutura MLflow - usar como exemplo runs SEGURO_NOVO_MANUAL v9.0.0 e v10.0.0
- Foco em:
    - transparência de desenvolvimento - código, dados, execuções e resultados em um único ambiente acessível ao cliente
    - implementação de regras de negócio e linhagem
    - análise de importância de features, escolha de modelos e parâmetros
    - análise de performance dos modelos e aderência da solução à operação
    - substrato para absorver e integrar o feedback do time comercial ao fluxo de desenvolvimento - O sistema atual é o substrato para abordagens progressivamente mais sofisticadas — novos dados, novas variáveis, novas técnicas de modelagem podem ser incorporados sem redesenhar a base

## Avaliação de resultados já obtidos
- Análises técnicas de performance dos modelos, à nível de desenvolvimento
- Análises executivas que justificam uso dos modelos como ferramenta de priorização

## Valor para time comercial
- Métricas baseadas em ranking e uso da solução como priorização de cotações
- Métricas que refletem o ganho no uso da ferramenta frente à alguma métrica de conversão atual do time comercial
- Métricas que relacionam capacidade operacional do time comercial (quantas cotações podem ser adereçadas em uma janela específica de tempo) - Esta etapa requer uma análise detalhada de capacidade operacional

# 4 - Estado atual de desenvolvimento e próximos passos

## Etapa atual
- Pipeline completo de treino operacional documentado no MLflow (ingestão → pré-processamento → modelo → scoring → comparação).
- Teste preliminares do pipeline e estrutura MLflow para segmentação SEGURO_NOVO_MANUAL, com análises de resultados já disponíveis

## Possibilidade de direcionamento
- Treinamento preliminar das demais segmentações (seguindo estratificação atual já mencionada)
- Definir e executar testes variando pré-processamento e parâmetros de treino -> Avaliação de relatórios de performance -> Nova iteração com área de negócio para direcionamento
- Desenvolvimento do pipeline de produção (Ingestão e tratamento dos dados submetidos à inferência de modelos treinados) -> Aprimorar arquitetura da solução e coletar feedbacks dos usuários de forma iterativa
- Análise de capacidade operacional para direcionar desenvolvimento
- Definição, caso necessário, de novas formas de análise

💡 Inserir visão de etapas já desenvolvidas e etapas faltantes