# Resumo desenvolvimento Intelligent Score Agent - AXA

## 1) Desafio e Solução - Interpretação da aplicação do sistema pelos times AXA/PSW

### Desafio
- Necessidade de abordagem data-driven para priorização de cotações de maior potencial de aprovação
- Limitação de operação do time comercial AXA

### Solução
- Desenvolvimento (Databricks + MLflow) de sistema de rankeamento de cotações - transparente, observável e escalável
- O desenvolvimento descrito (a nível de aplicação/objetivo, modelos, fluxo, análises) pode e deve sofrer alterações a partir de interações com área executiva e time comercial. Hoje, o pipeline contempla modelos tradicionais de classificação (simples e interpretáveis). 

#### Diagrama pipelines de treinamento e inferência:
![alt text](image.png)

- Pipeline de treinamento + observabilidade via MLflow (já implementado)
    - Contempla logging de etapas desde a ingestão dos dados no ambiente Databricks até pré-processamento/regras de negócio aplicadas, treinamento e análise de performance e resultados. Esta abordagem possibilita desenvolver e comparar modelagens com transparência e velocidade, além de monitorar o comportamento dos modelos com dados reais. Hoje, a equipe AXA tem acesso ao ambiente Databricks para consulta do desenvolvimento (psw-databricks-axa).
- Pipeline de inferência/distribuição + observabilidade via MLflow (pendente implementação)

A partir de iterações através de agendas pontuais, os resultados disponibilizados nas execuções do pipeline podem ser acoplados aos feedbacks recebidos dos usuários, de modo a direcionar próximos testes/versões.

#### Segmentações
- Hoje, existem 4 segmentações
    - SEGURO_NOVO_DIGITAL
    - SEGURO_NOVO_MANUAL
    - RENOVACAO_DIGITAL
    - RENOVACAO_MANUAL
- Idealmente, é desenvolvido um modelo por segmentações, podendo ser criados mais modelos (estratificados por produtos, por exemplo), acomodados na estrutura já criada, que confere velocidade de desenvolvimento e observabilidade para AXA.

## Etapa atual
- Pipeline completo de treino operacional documentado no MLflow (ingestão → pré-processamento → modelo → scoring → comparação).
- Teste preliminares do pipeline e estrutura MLflow para segmentação SEGURO_NOVO_MANUAL, com análises de resultados já disponíveis

## Possibilidade de direcionamento
- Treinamento preliminar das demais segmentações (seguindo estratificação atual já mencionada)
- Definir e executar testes variando pré-processamento e parâmetros de treino -> Avaliação de relatórios de performance -> Nova iteração com área de negócio para direcionamento
- Desenvolvimento do pipeline de produção (Ingestão e tratamento dos dados submetidos à inferência de modelos treinados) -> Aprimorar arquitetura da solução e coletar feedbacks dos usuários de forma iterativa
- Análise de capacidade operacional para direcionar desenvolvimento
- Definição, caso necessário, de novas formas de análise


