# Panorama de Padroes de Desenvolvimento — AXA ISA Workspace

**Data:** 2026-03-25 | **Projeto:** Pipeline ML de scoring/ranking de cotacoes (Databricks/PySpark)
**Stack:** Python, PySpark, MLflow, Delta Lake, Databricks Notebooks

---

## O QUE TEMOS (padroes existentes)

| Categoria | Detalhes |
|---|---|
| **Arquitetura de dados** | Medallion (bronze → silver → gold) com naming convention `<schema>.<tabela>_<TS_EXEC>` |
| **Experiment tracking** | MLflow robusto — hierarquia parent/child runs, params, metricas, artifacts, tags, profiling, lineage JSON |
| **Pipeline sequencial** | 6 notebooks bem definidos (0→5), cada um com I/O documentado e etapa clara |
| **Rule engine** | Sistema declarativo `rule_def` + `apply_rules` com toggles habilitaveis/desabilitaveis por regra |
| **Feature management** | Dict `FEATURE_CANDIDATES` com toggles True/False por feature, inferencia automatica de tipo |
| **Versionamento semantico** | VX.Y.Z nos notebooks de treino/inferencia/comparacao (atualmente V10.0.0) |
| **Secrets** | `dbutils.secrets.get()` para credenciais SFTP — nao expostos em codigo |
| **Documentacao** | CLAUDE.md (guia estrutural), BACKLOG.md, LINEAGE_TABELAS_MANUAL.md, REFS_MODELAGEM.md, ANALISE_V11.md |
| **Diagramas** | draw.io com visao de pipeline (v1, v2) |
| **Reprodutibilidade** | Seeds, salts e parametros logados no MLflow; referencia cruzada entre runs via run_id |
| **Git** | Repositorio GitHub com remote configurado, `.gitignore` presente |
| **Profiling** | `profiling.json` + graficos como artifacts MLflow |

---

## O QUE FALTA (lacunas identificadas)

### Prioridade Alta

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem testes automatizados** | Nenhum unit/integration test. Regressoes passam despercebidas | Criar testes para funcoes utilitarias e regras (pytest) |
| **Sem CI/CD** | Nenhum GitHub Actions ou pipeline automatizado. Deploys manuais | Minimo: lint + testes no push. Ideal: validacao de notebooks |
| **Sem orquestracao** | Notebooks executados manualmente, params preenchidos a mao entre etapas | Databricks Workflows ou Jobs encadeados |
| **Codigo duplicado entre notebooks** | `ensure_schema`, `table_exists`, `mlflow_get_or_create_experiment`, `profile_basic` replicados em 3+ notebooks | Extrair para modulo compartilhado (`utils.py` ou wheel) |
| **Sem gerenciamento de dependencias** | Nenhum `requirements.txt`, `pyproject.toml` ou lock file | Documentar dependencias com versoes pinadas |
| **Commits genericos** | 100% dos commits sao "atualizacao" / "atualizacoes" | Adotar Conventional Commits (`feat:`, `fix:`, `docs:`) |
| **Token PAT exposto no git remote** | Credencial em plaintext na config do git | Revogar imediatamente, migrar para SSH ou credential helper |

### Prioridade Media

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem branch strategy** | Apenas branch `main`, sem feature branches ou PRs | Git Flow simplificado: `main` + feature branches + PRs com review |
| **Sem linting/formatting** | Sem Black, Ruff, Flake8 — inconsistencia de estilo | Configurar Ruff ou Black + isort com pre-commit |
| **Sem validacao de dados automatizada** | Sem gates entre etapas (null checks, schema validation, row count assertions) | Great Expectations ou validacoes minimas no notebook |
| **Error handling inconsistente** | Bare `except:` sem tipo em multiplos notebooks | Capturar excecoes especificas, logging estruturado |
| **Sem logging estruturado** | Apenas `print()` para rastreamento — impossivel filtrar/buscar | Usar `logging` module ou integrar com Databricks log analytics |
| **Sem Model Registry** | Modelos nao registrados no MLflow Model Registry — sem staging/prod | Registrar modelos com estagio (Staging → Production) |
| **Configuracao hardcoded** | Params em celulas de notebook, editados manualmente a cada execucao | Externalizar configs (YAML/JSON) ou Databricks Widgets |

### Prioridade Baixa (melhorias futuras)

| Lacuna | Impacto | Recomendacao |
|---|---|---|
| **Sem README no repositorio** | Onboarding dificil para novos devs | README.md com setup, pre-requisitos e fluxo |
| **Sem pre-commit hooks** | Nada impede commits com erros de lint ou secrets | Configurar pre-commit (ruff, detect-secrets) |
| **Lineage apenas manual** | `LINEAGE_TABELAS_MANUAL.md` desatualiza facilmente | Automatizar via MLflow artifacts + script de consolidacao |
| **Sem monitoramento de pipeline** | Sem alertas de falha ou drift | Integrar alertas (Slack/email) em Databricks Jobs |
| **Sem Feature Store** | Features recalculadas a cada execucao | Avaliar Databricks Feature Store para consistencia treino↔inferencia |

---

## METRICAS DE MATURIDADE

| Dimensao | Nivel (1-5) | Observacao |
|---|---|---|
| **Experiment tracking** | 4/5 | Maduro — MLflow bem estruturado, falta Model Registry |
| **Arquitetura de dados** | 4/5 | Medallion clara, lineage documentada, falta automacao |
| **Qualidade de codigo** | 2/5 | Sem testes, lint, ou modularizacao |
| **DevOps/CI-CD** | 1/5 | Inexistente |
| **Versionamento** | 2/5 | Git presente, mas sem branching, PRs, ou commits descritivos |
| **Documentacao** | 3/5 | Boa para o tamanho do projeto, falta README e docs automatizadas |
| **Seguranca** | 2/5 | Secrets ok no codigo, mas PAT exposto no git remote |
| **Operacao/Producao** | 1/5 | Pipeline 100% manual, sem orquestracao nem monitoramento |

---

## RECOMENDACAO DE ROADMAP (ordem sugerida)

1. **Revogar token PAT** do git remote (imediato)
2. **Extrair funcoes compartilhadas** para `utils.py`
3. **Adicionar `requirements.txt`** com dependencias pinadas
4. **Adotar Conventional Commits** + branch strategy minima
5. **Configurar GitHub Actions** com lint basico (Ruff)
6. **Implementar testes** para regras e funcoes utilitarias
7. **Configurar Databricks Workflows** para orquestracao
8. **Registrar modelos** no MLflow Model Registry
