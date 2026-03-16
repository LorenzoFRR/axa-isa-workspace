# Databricks notebook source
# MAGIC %md
# MAGIC ## Configs

# COMMAND ----------

from datetime import datetime
from zoneinfo import ZoneInfo
import uuid

# =========================
# MLflow (estrutura)
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"  # <<< AJUSTE

PR_TREINO_NAME  = "T_PR_TREINO"
MODE_CODE       = "C"
MODE_NAME       = f"T_MODE_{MODE_CODE}"

PR_RUN_ID_OVERRIDE        = "21f1184afb354159b325ad87bca3c50d"
MODE_RUN_ID_OVERRIDE      = "71c8526241ab41ad88bd585679d1e9e5"
PRE_PROC_RUN_ID_OVERRIDE  = "0c3c06a104264ba2aee0a8a2ec032edd"
FS_RUN_ID_OVERRIDE        = "0ae2a9f6d8b240fdb835858a36592a66"
TREINO_RUN_ID_OVERRIDE    = "0c43caa200884d9c89ddb0ca0469c65a"

STEP_PRE_PROC_NAME          = "T_PRE_PROC_MODEL"
STEP_FEATURE_SELECTION_NAME = "T_FEATURE_SELECTION"
STEP_TREINO_NAME            = "T_TREINO"

# =========================
# Versionamento
# =========================
TREINO_VERSAO            = "V9.0.0"
TREINO_VERSAO_TABLE_SAFE = TREINO_VERSAO.replace(".", "_")
VERSAO_REF               = TREINO_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_UUID   = uuid.uuid4().hex[:8]
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# =========================
# INPUT
# =========================
COTACAO_SEG_FQN = "silver.cotacao_seg_20260310_163401"  # <<< AJUSTE

# =========================
# OUTPUT
# =========================
OUT_SCHEMA   = "gold"
DF_MODEL_FQN = f"{OUT_SCHEMA}.cotacao_model_{TS_EXEC}_{RUN_UUID}"
DF_VALID_FQN = f"{OUT_SCHEMA}.cotacao_validacao_{TS_EXEC}_{RUN_UUID}"
WRITE_MODE   = "overwrite"

# =========================
# PRE_PROC_MODEL — params
# =========================
STATUS_COL           = "DS_GRUPO_STATUS"
LABEL_COL            = "label"
ID_COL               = "CD_NUMERO_COTACAO_AXA"
SEG_COL              = "SEG"
DATE_COL             = "DATA_COTACAO"
ALLOWED_FINAL_STATUS = ["Emitida", "Perdida"]
VALID_FRAC           = 0.20
SPLIT_SALT           = "split_c1_seg_mes"  # <<< AJUSTE — string auditável

DO_PROFILE = True

# =========================
# PRE_PROC_MODEL / FS — colunas e segmento
# =========================
# SEG_TARGET é usado no PRE_PROC_MODEL (logado como param) e reutilizado no FS e TREINO.
SEG_TARGET = "SEGURO_NOVO_MANUAL"  # <<< AJUSTE

FS_SEEDS            = [42, 123, 7]
FS_TRAIN_FRAC       = 0.70
NULL_DROP_PCT       = 0.90
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# Colunas excluídas estruturalmente — nunca serão features independente de toggles.
COLS_NEVER_FEATURE = [
    ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL,
    STATUS_COL, LABEL_COL, "MES",
]

# Toggles de features — True = entra no pipeline | False = bloqueada
# Tipo (cat/num) é inferido automaticamente do schema — sem necessidade de classificar aqui.
# Executar a célula de inspeção abaixo para gerar este dict a partir da tabela.
FEATURE_CANDIDATES = {
    "VL_PREMIO_ALVO":                            True,   # decimal(17,2)
    "INTERMENDIARIO_PERFIL":                     True,   # string
    "DT_INICIO_VIGENCIA":                        False,   # date
    "VL_PREMIO_LIQUIDO":                         True,   # decimal(17,2)
    "VL_PRE_TOTAL":                              True,   # decimal(17,2)
    "DS_PRODUTO_NOME":                           True,   # string
    "DS_SISTEMA":                                True,   # string
    "VL_ENDOSSO_PREMIO_TOTAL":                   True,   # decimal(17,2)
    "CD_FILIAL_RESPONSAVEL_COTACAO":             True,   # string
    "DS_ATIVIDADE_SEGURADO":                     True,   # string
    "DS_GRUPO_CORRETOR_SEGMENTO":                True,   # string
    "DIAS_ULTIMA_ATUALIZACAO":                   False,   # int
    "DIAS_VALIDADE":                             False,   # int
    "DIAS_ANALISE_SUBSCRICAO":                   False,   # int
    "DIAS_FIM_ANALISE_SUBSCRICAO":               False,   # int
    "DIAS_COTACAO":                              False,   # int
    "DIAS_INICIO_VIGENCIA":                      False,   # int
    "VL_GWP_CORRETOR_resumo":                    True,   # decimal(17,2)
    "QTD_ACORDO_COMERCIAL_resumo":               False,   # string
    "QTD_COTACAO_2024_detalhe":                  False,   # string
    "QTD_COTACAO_2025_detalhe":                  False,   # string
    "QTD_COTACAO_M2_detalhe":                    False,   # string
    "QTD_COTACAO_M3_detalhe":                    False,   # string
    "QTD_EMITIDO_2024_detalhe":                  False,   # string
    "QTD_EMITIDO_2025_detalhe":                  False,   # string
    "QTD_EMITIDO_M2_detalhe":                    False,   # string
    "QTD_EMITIDO_M3_detalhe":                    False,   # string
    "HR_2024_detalhe":                           False,   # decimal(17,6)
    "HR_2025_detalhe":                           False,   # decimal(17,6)
    "HR_M2_detalhe":                             False,   # decimal(17,6)
    "HR_M3_detalhe":                             False,   # decimal(17,6)
}


'VL_PREMIO_ALVO', 'VL_ENDOSSO_PREMIO_TOTAL'


FS_METHODS_CONFIG = {
    "lr_l1": {"maxIter": 100, "regParam": 0.01, "elasticNetParam": 1.0},
    "rf":    {"numTrees": 200, "maxDepth": 8},
    "gbt":   {"maxIter": 80,  "maxDepth": 5, "stepSize": 0.1},
}

TOPK_LIST = [5]

print("✅ CONFIG MODE_C carregada")
print("• input         :", COTACAO_SEG_FQN)
print("• mode          :", MODE_CODE)
print("• versao        :", TREINO_VERSAO)
print("• seg_target    :", SEG_TARGET)
print("• split_salt    :", SPLIT_SALT)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspeção de colunas candidatas
# MAGIC Executar esta célula para gerar `FEATURE_CANDIDATES` a partir da tabela.
# MAGIC Copie o output na Config acima e ajuste os toggles antes de rodar o pipeline.

# COMMAND ----------

_df_inspect = spark.table(COTACAO_SEG_FQN)
_schema_map = dict(_df_inspect.dtypes)
_never_feat = set(COLS_NEVER_FEATURE)
_candidates = [c for c in _df_inspect.columns if c not in _never_feat]

print("# ─── Cole em FEATURE_CANDIDATES na Config ──────────────────────────────")
print("FEATURE_CANDIDATES = {")
for _c in _candidates:
    _t   = _schema_map[_c]
    _pad = " " * max(1, 42 - len(_c))
    print(f'    "{_c}":{_pad}True,   # {_t}')
print("}")
print()
print(f"# Candidatas: {len(_candidates)} | Excluídas estruturalmente ({len(_never_feat)}): {sorted(_never_feat)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers globais

# COMMAND ----------

import json
from typing import Callable, Dict, List, Optional, Tuple

import mlflow

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


def ensure_schema(schema_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

def table_exists(fqn: str) -> bool:
    return spark.catalog.tableExists(fqn)

def assert_table_exists(fqn: str) -> None:
    if not table_exists(fqn):
        raise ValueError(f"❌ Tabela não existe: {fqn}")

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id

def safe_drop_cols(df: DataFrame, cols: List[str]) -> DataFrame:
    existing = set(df.columns)
    to_drop  = [c for c in cols if c in existing]
    return df.drop(*to_drop) if to_drop else df

RuleFn = Callable[[DataFrame], DataFrame]

def rule_def(rule_id, description, fn, enabled=True, requires_columns=None):
    return {
        "rule_id": rule_id, "description": description,
        "enabled": enabled, "requires_columns": requires_columns or [], "fn": fn,
    }

def apply_rules_block(block_key, df, rules, enable_rules=True, toggles=None):
    toggles  = toggles or {}
    df_out   = df
    exec_log = []
    for r in rules:
        rid, desc, req = r["rule_id"], r["description"], r.get("requires_columns", []) or []
        enabled = enable_rules and bool(r.get("enabled", True))
        if block_key in toggles and rid in toggles[block_key]:
            enabled = enabled and bool(toggles[block_key][rid])
        if not enabled:
            exec_log.append({"rule_id": rid, "status": "SKIPPED_DISABLED", "description": desc})
            continue
        missing = [c for c in req if c not in df_out.columns]
        if missing:
            exec_log.append({"rule_id": rid, "status": "SKIPPED_MISSING_COLS", "description": desc, "reason": f"missing={missing}"})
            continue
        try:
            df_out = r["fn"](df_out)
            exec_log.append({"rule_id": rid, "status": "APPLIED", "description": desc})
        except Exception as e:
            raise RuntimeError(f"❌ Falha regra {block_key}.{rid}: {e}") from e
    return df_out, exec_log

def rules_catalog_for_logging(rules_by_block):
    return {
        blk: [{"rule_id": r["rule_id"], "description": r["description"],
               "enabled": bool(r.get("enabled", True)),
               "requires_columns": r.get("requires_columns", [])}
              for r in rules]
        for blk, rules in rules_by_block.items()
    }

def profile_basic(df: DataFrame, name: str, key_cols=None):
    n_rows = df.count()
    exprs  = [
        F.sum(F.when(F.col(c).isNull() | ((t == "string") & (F.trim(F.col(c)) == "")), 1).otherwise(0)).alias(c)
        for c, t in df.dtypes
    ]
    nulls = df.agg(*exprs).collect()[0].asDict()
    out   = {"name": name, "n_rows": int(n_rows), "n_cols": len(df.columns),
             "null_count": {k: int(v) for k, v in nulls.items()}}
    if key_cols:
        out["distinct_count"] = {
            k: int(df.select(k).distinct().count())
            for k in key_cols if k in df.columns
        }
    return out

def counts_by_seg(df, seg_col):
    if seg_col not in df.columns:
        return []
    rows = df.groupBy(seg_col).count().orderBy(F.col("count").desc()).collect()
    return [{seg_col: r[seg_col], "count": int(r["count"])} for r in rows]

def label_rate_by_seg(df, seg_col, label_col):
    if seg_col not in df.columns or label_col not in df.columns:
        return []
    rows = (df.groupBy(seg_col)
              .agg(F.count(F.lit(1)).alias("n"), F.avg(F.col(label_col).cast("double")).alias("label_rate"))
              .orderBy(F.col("n").desc()).collect())
    return [{seg_col: r[seg_col], "n": int(r["n"]), "label_rate": float(r["label_rate"])} for r in rows]


# =========================
# Lineage helpers
# =========================
def build_tables_lineage_preproc() -> dict:
    return {
        "stage":         "T_PRE_PROC_MODEL",
        "ts_exec":       TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode":          MODE_CODE,
        "inputs":        {"cotacao_seg": COTACAO_SEG_FQN},
        "outputs":       {"df_model": DF_MODEL_FQN, "df_validacao": DF_VALID_FQN},
    }

def build_tables_lineage_fs() -> dict:
    return {
        "stage":         "T_FEATURE_SELECTION",
        "ts_exec":       TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode":          MODE_CODE,
        "inputs":        {"df_model": DF_MODEL_FQN},
        "seg_target":    SEG_TARGET,
    }


def compute_top_vals(df: DataFrame, col_name: str, top_n: int) -> list:
    """
    Calcula os top_n valores mais frequentes de col_name (excluindo nulos/brancos).
    Retorna lista de valores — usado para truncagem de cardinalidade rastreável.
    """
    return (
        df.filter(F.col(col_name).isNotNull() & (F.trim(F.col(col_name)) != ""))
          .groupBy(col_name).count()
          .orderBy(F.col("count").desc())
          .limit(top_n)
          .select(col_name)
          .rdd.flatMap(lambda x: x)
          .collect()
    )


def apply_truncation(df: DataFrame, col_name: str, top_vals: list, outros: str = "OUTROS") -> DataFrame:
    """
    Substitui valores fora de top_vals (incluindo nulos/brancos) por outros.
    Separado de compute_top_vals para reutilizar top_vals_by_col salvo no artifact store.
    """
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(top_vals), F.col(col_name)).otherwise(F.lit(outros)),
    )


print("✅ Helpers globais carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_PRE_PROC_MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funções de regra

# COMMAND ----------

def PP_R01_normaliza_status(df: DataFrame) -> DataFrame:
    s = F.upper(F.trim(F.col(STATUS_COL).cast("string")))
    return df.withColumn(STATUS_COL,
        F.when(s == "EMITIDA", F.lit("Emitida"))
         .when(s == "PERDIDA", F.lit("Perdida"))
         .otherwise(F.col(STATUS_COL)))

def PP_R02_filtra_status_finais(df: DataFrame) -> DataFrame:
    # MODE_C: sem conversão de status intermediários para Perdida.
    # Filtrar apenas, mantendo somente Emitida e Perdida.
    return df.filter(F.col(STATUS_COL).isin(ALLOWED_FINAL_STATUS))

def PP_R03_cria_label(df: DataFrame) -> DataFrame:
    return df.withColumn(LABEL_COL,
        F.when(F.col(STATUS_COL) == "Emitida", F.lit(1.0))
         .when(F.col(STATUS_COL) == "Perdida", F.lit(0.0))
         .otherwise(F.lit(None).cast("double")))

def BUILD_R01_add_mes(df: DataFrame) -> DataFrame:
    return (df.withColumn("DATA_COTACAO_dt", F.to_date(F.col(DATE_COL)))
              .filter(F.col("DATA_COTACAO_dt").isNotNull())
              .withColumn("MES", F.date_format("DATA_COTACAO_dt", "yyyy-MM")))

def BUILD_R02_add_split_flag(df: DataFrame) -> DataFrame:
    h = F.xxhash64(
        F.col(ID_COL).cast("string"), F.col(SEG_COL).cast("string"),
        F.col("MES").cast("string"), F.lit(SPLIT_SALT),
    )
    score = F.pmod(F.abs(h), F.lit(1_000_000)) / F.lit(1_000_000.0)
    return (df.withColumn("_split_score", score)
              .withColumn("is_valid", F.col("_split_score") < F.lit(float(VALID_FRAC))))

def MODEL_R03_filtra_model(df: DataFrame) -> DataFrame:
    return df.filter(F.col("is_valid") == F.lit(False))

def VALID_R03_filtra_validacao(df: DataFrame) -> DataFrame:
    return df.filter(F.col("is_valid") == F.lit(True))

def BUILD_R04_drop_aux(df: DataFrame) -> DataFrame:
    return safe_drop_cols(df, ["DATA_COTACAO_dt", "_split_score", "is_valid"])

print("✅ Funções de regra T_PRE_PROC_MODEL carregadas")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Toggles — T_PRE_PROC_MODEL

# COMMAND ----------

# True = regra ativa | False = regra desabilitada
TOGGLES_RULES_ON_DF_SEG = {
    "PP_R01": True,   # Normalizar DS_GRUPO_STATUS (EMITIDA→Emitida, PERDIDA→Perdida)
    "PP_R02": True,   # Manter apenas status finais (Emitida, Perdida) — sem converter intermediários
    "PP_R03": True,   # Criar coluna label (Emitida=1.0, Perdida=0.0)
}

TOGGLES_RULES_BUILD_BASE = {
    "BUILD_R01": True,  # Criar MES=yyyy-MM a partir de DATA_COTACAO (filtra DATA_COTACAO nula)
    "BUILD_R02": True,  # Criar split determinístico por (ID, SEG, MES, salt) → is_valid
}

TOGGLES_RULES_BUILD_MODEL = {
    "MODEL_R03": True,  # Selecionar df_model (is_valid=False)
    "BUILD_R04": True,  # Remover colunas auxiliares do split
}

TOGGLES_RULES_BUILD_VALID = {
    "VALID_R03": True,  # Selecionar df_validacao (is_valid=True)
    "BUILD_R04": True,  # Remover colunas auxiliares do split
}

TOGGLES_RULES_FEATURE_PREP = {
    "PP_R04": True,   # Remoção de features com >NULL_DROP_PCT nulos
    "PP_R05": True,   # Truncagem de alta cardinalidade (>HIGH_CARD_THRESHOLD → top HIGH_CARD_TOP_N + OUTROS)
    "PP_R06": True,   # Remoção de features constantes (cardinalidade <= 1)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Catálogo de regras — T_PRE_PROC_MODEL

# COMMAND ----------

RULES_ON_DF_SEG = [
    rule_def("PP_R01", "Normalizar DS_GRUPO_STATUS (EMITIDA→Emitida, PERDIDA→Perdida)",
             PP_R01_normaliza_status,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R01"],
             requires_columns=[STATUS_COL]),
    rule_def("PP_R02", "Manter apenas status finais (Emitida, Perdida) — sem converter intermediários",
             PP_R02_filtra_status_finais,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R02"],
             requires_columns=[STATUS_COL]),
    rule_def("PP_R03", "Criar label (Emitida=1.0, Perdida=0.0)",
             PP_R03_cria_label,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R03"],
             requires_columns=[STATUS_COL]),
]

RULES_BUILD_BASE = [
    rule_def("BUILD_R01", "Criar MES=yyyy-MM a partir de DATA_COTACAO",
             BUILD_R01_add_mes,
             enabled=TOGGLES_RULES_BUILD_BASE["BUILD_R01"],
             requires_columns=[DATE_COL]),
    rule_def("BUILD_R02", "Criar split determinístico por (ID, SEG, MES, salt)",
             BUILD_R02_add_split_flag,
             enabled=TOGGLES_RULES_BUILD_BASE["BUILD_R02"],
             requires_columns=[ID_COL, SEG_COL]),
]

RULES_BUILD_MODEL = [
    rule_def("MODEL_R03", "Selecionar df_model (is_valid=False)",
             MODEL_R03_filtra_model,
             enabled=TOGGLES_RULES_BUILD_MODEL["MODEL_R03"],
             requires_columns=["is_valid"]),
    rule_def("BUILD_R04", "Remover colunas auxiliares do split",
             BUILD_R04_drop_aux,
             enabled=TOGGLES_RULES_BUILD_MODEL["BUILD_R04"]),
]

RULES_BUILD_VALID = [
    rule_def("VALID_R03", "Selecionar df_validacao (is_valid=True)",
             VALID_R03_filtra_validacao,
             enabled=TOGGLES_RULES_BUILD_VALID["VALID_R03"],
             requires_columns=["is_valid"]),
    rule_def("BUILD_R04", "Remover colunas auxiliares do split",
             BUILD_R04_drop_aux,
             enabled=TOGGLES_RULES_BUILD_VALID["BUILD_R04"]),
]

RULES_FEATURE_PREP = [
    rule_def("PP_R04",
             f"Remoção de features com >{int(NULL_DROP_PCT * 100)}% nulos (null_drop_pct={NULL_DROP_PCT})",
             lambda df: df,
             enabled=TOGGLES_RULES_FEATURE_PREP["PP_R04"]),
    rule_def("PP_R05",
             f"Truncagem de alta cardinalidade: >{HIGH_CARD_THRESHOLD} categorias → top {HIGH_CARD_TOP_N} + '{OUTROS_LABEL}'",
             lambda df: df,
             enabled=TOGGLES_RULES_FEATURE_PREP["PP_R05"]),
    rule_def("PP_R06",
             "Remoção de features constantes (cardinalidade <= 1)",
             lambda df: df,
             enabled=TOGGLES_RULES_FEATURE_PREP["PP_R06"]),
]

RULES_BY_BLOCK = {
    "rules_on_df_seg":          RULES_ON_DF_SEG,
    "rules_build_base":         RULES_BUILD_BASE,
    "rules_build_df_model":     RULES_BUILD_MODEL,
    "rules_build_df_validacao": RULES_BUILD_VALID,
    "rules_feature_prep":       RULES_FEATURE_PREP,
}

print("✅ RULES_BY_BLOCK definido")
print("• rules_on_df_seg         :", len(RULES_ON_DF_SEG))
print("• rules_build_base        :", len(RULES_BUILD_BASE))
print("• rules_build_df_model    :", len(RULES_BUILD_MODEL))
print("• rules_build_df_validacao:", len(RULES_BUILD_VALID))
print("• rules_feature_prep      :", len(RULES_FEATURE_PREP))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução — T_PRE_PROC_MODEL

# COMMAND ----------

ensure_schema(OUT_SCHEMA)
assert_table_exists(COTACAO_SEG_FQN)
if table_exists(DF_MODEL_FQN):
    raise ValueError(f"❌ DF_MODEL já existe: {DF_MODEL_FQN}")
if table_exists(DF_VALID_FQN):
    raise ValueError(f"❌ DF_VALID já existe: {DF_VALID_FQN}")

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
while mlflow.active_run() is not None:
    mlflow.end_run()

if PR_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    PR_RUN_ID = PR_RUN_ID_OVERRIDE
else:
    mlflow.start_run(run_name=PR_TREINO_NAME)
    PR_RUN_ID = mlflow.active_run().info.run_id

if MODE_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=MODE_RUN_ID_OVERRIDE, nested=True)
    MODE_RUN_ID = MODE_RUN_ID_OVERRIDE
else:
    mlflow.start_run(run_name=MODE_NAME, nested=True)
    MODE_RUN_ID = mlflow.active_run().info.run_id

RUN_PRE_PROC_EXEC = run_name_vts("T_PRE_PROC_MODEL")
_pp_kw = {"run_id": PRE_PROC_RUN_ID_OVERRIDE} if PRE_PROC_RUN_ID_OVERRIDE else {"run_name": STEP_PRE_PROC_NAME}

with mlflow.start_run(**_pp_kw, nested=True) as pp_container:
    PREPROC_CONTAINER_RUN_ID = pp_container.info.run_id

    with mlflow.start_run(run_name=RUN_PRE_PROC_EXEC, nested=True):
        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "PRE_PROC_MODEL",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })
        mlflow.log_params({
            "ts_exec": TS_EXEC, "treino_versao": TREINO_VERSAO, "mode_code": MODE_CODE,
            "seg_target": SEG_TARGET,
            "input_cotacao_seg_fqn": COTACAO_SEG_FQN,
            "df_model_fqn": DF_MODEL_FQN, "df_validacao_fqn": DF_VALID_FQN,
            "valid_frac": float(VALID_FRAC), "split_salt": SPLIT_SALT,
            "allowed_final_status": json.dumps(ALLOWED_FINAL_STATUS),
            "label_col": LABEL_COL, "id_col": ID_COL, "seg_col": SEG_COL,
            "note_pp_r02": "MODE_C: sem conversao de intermediarios para Perdida",
            "pr_run_id": PR_RUN_ID, "mode_run_id": MODE_RUN_ID,
            "t_pre_proc_model_container_run_id": PREPROC_CONTAINER_RUN_ID,
        })
        mlflow.log_dict(rules_catalog_for_logging(RULES_BY_BLOCK), "rules_catalog.json")
        mlflow.log_dict(build_tables_lineage_preproc(), "tables_lineage.json")

        df_seg_in = spark.table(COTACAO_SEG_FQN)
        n_seg_in  = int(df_seg_in.count())
        mlflow.log_metric("n_seg_in", n_seg_in)

        # ── rules_on_df_seg — aplicar individualmente para logar n_linhas_por_regra ──
        df_seg_pp   = df_seg_in
        exec_log    = {}
        exec_log_seg = []
        for r in RULES_ON_DF_SEG:
            rid, desc = r["rule_id"], r["description"]
            req       = r.get("requires_columns", []) or []
            is_enabled = bool(r.get("enabled", True)) and bool(TOGGLES_RULES_ON_DF_SEG.get(rid, True))
            if not is_enabled:
                exec_log_seg.append({"rule_id": rid, "status": "SKIPPED_DISABLED", "description": desc})
                continue
            missing = [c for c in req if c not in df_seg_pp.columns]
            if missing:
                exec_log_seg.append({"rule_id": rid, "status": "SKIPPED_MISSING_COLS", "description": desc,
                                     "reason": f"missing={missing}"})
                continue
            df_seg_pp  = r["fn"](df_seg_pp)
            n_after    = int(df_seg_pp.count())
            mlflow.log_metric(f"n_linhas_por_regra_{rid}", n_after)
            exec_log_seg.append({"rule_id": rid, "status": "APPLIED", "description": desc,
                                  "n_rows_after": n_after})

        exec_log["rules_on_df_seg"] = exec_log_seg
        mlflow.log_metric("n_seg_after_rules", int(df_seg_pp.count()))

        df_base, exec_log["rules_build_base"]             = apply_rules_block("rules_build_base", df_seg_pp, RULES_BUILD_BASE)
        df_model_tmp, exec_log["rules_build_df_model"]     = apply_rules_block("rules_build_df_model", df_base, RULES_BUILD_MODEL)
        df_valid_tmp, exec_log["rules_build_df_validacao"] = apply_rules_block("rules_build_df_validacao", df_base, RULES_BUILD_VALID)

        # ── PP_R04 / PP_R05 / PP_R06 — feature prep aplicado antes de salvar ────
        # Centralizado aqui para que df_model já salvo reflita todos os tratamentos.
        # top_vals_by_col_prep fica disponível como variável Python para T_TREINO.
        _schema_pp      = dict(df_model_tmp.dtypes)
        _fc_all_enabled = [c for c, v in FEATURE_CANDIDATES.items() if v and c in _schema_pp]
        _pp_cat_cols    = [c for c in _fc_all_enabled if _schema_pp[c] == "string"]
        _pp_num_cols    = [c for c in _fc_all_enabled if _schema_pp[c] != "string"]
        _pp_all_cand    = _pp_cat_cols + _pp_num_cols

        n_rows_pp = int(df_model_tmp.count())

        # PP_R04 — remoção de features com >NULL_DROP_PCT nulos
        null_counts_pp  = df_model_tmp.agg(*[
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in _pp_all_cand
        ]).collect()[0].asDict()
        null_profile_pp = [
            {"col": c, "nulls": int(null_counts_pp[c]),
             "pct_null": round(null_counts_pp[c] / n_rows_pp, 4)}
            for c in _pp_all_cand
        ]
        if TOGGLES_RULES_FEATURE_PREP["PP_R04"]:
            drop_null_cols_pp = sorted({r["col"] for r in null_profile_pp if r["pct_null"] > NULL_DROP_PCT})
            if drop_null_cols_pp:
                df_model_tmp = df_model_tmp.drop(*drop_null_cols_pp)
                df_valid_tmp = df_valid_tmp.drop(*drop_null_cols_pp)
            exec_log["PP_R04"] = {"status": "APPLIED", "drop_null_cols": drop_null_cols_pp}
        else:
            drop_null_cols_pp = []
            exec_log["PP_R04"] = {"status": "SKIPPED_DISABLED"}

        # PP_R05 — truncagem de alta cardinalidade
        cat_present_pp  = [c for c in _pp_cat_cols if c in df_model_tmp.columns]
        cat_card_pre_pp = [
            {"col": c, "count_distinct": int(df_model_tmp.select(F.countDistinct(c)).collect()[0][0])}
            for c in cat_present_pp
        ]
        high_card_cols_pp    = [r["col"] for r in cat_card_pre_pp if r["count_distinct"] > HIGH_CARD_THRESHOLD]
        top_vals_by_col_prep: dict = {}
        if TOGGLES_RULES_FEATURE_PREP["PP_R05"]:
            for c in high_card_cols_pp:
                top_vals_by_col_prep[c] = compute_top_vals(df_model_tmp, c, HIGH_CARD_TOP_N)
                df_model_tmp = apply_truncation(df_model_tmp, c, top_vals_by_col_prep[c], OUTROS_LABEL)
                df_valid_tmp = apply_truncation(df_valid_tmp, c, top_vals_by_col_prep[c], OUTROS_LABEL)
            exec_log["PP_R05"] = {"status": "APPLIED", "high_card_cols": high_card_cols_pp}
        else:
            exec_log["PP_R05"] = {"status": "SKIPPED_DISABLED"}

        # PP_R06 — remoção de features constantes
        cat_for_const_pp = [c for c in cat_present_pp if c in df_model_tmp.columns]
        if TOGGLES_RULES_FEATURE_PREP["PP_R06"]:
            drop_const_cols_pp = sorted({
                c for c in cat_for_const_pp
                if int(df_model_tmp.select(F.countDistinct(c)).collect()[0][0]) <= 1
            })
            if drop_const_cols_pp:
                df_model_tmp = df_model_tmp.drop(*drop_const_cols_pp)
                df_valid_tmp = df_valid_tmp.drop(*drop_const_cols_pp)
            exec_log["PP_R06"] = {"status": "APPLIED", "drop_constant_cols": drop_const_cols_pp}
        else:
            drop_const_cols_pp = []
            exec_log["PP_R06"] = {"status": "SKIPPED_DISABLED"}

        mlflow.log_params({
            "null_drop_pct":       NULL_DROP_PCT,
            "high_card_threshold": HIGH_CARD_THRESHOLD,
            "high_card_top_n":     HIGH_CARD_TOP_N,
            "outros_label":        OUTROS_LABEL,
        })
        mlflow.log_dict(
            {"null_profile": null_profile_pp, "drop_null_cols": drop_null_cols_pp},
            "preproc_feature/null_profile.json",
        )
        mlflow.log_dict(
            {
                "cat_cardinality_pre": cat_card_pre_pp,
                "high_card_cols":      high_card_cols_pp,
                "top_n":               HIGH_CARD_TOP_N,
                "top_vals_by_col":     top_vals_by_col_prep,
            },
            "preproc_feature/cat_cardinality.json",
        )
        mlflow.log_dict(
            rules_catalog_for_logging({"rules_feature_prep": RULES_FEATURE_PREP}),
            "preproc_feature/rules_feature_prep_catalog.json",
        )
        mlflow.log_dict(exec_log, "rules_execution.json")

        n_model = int(df_model_tmp.count())
        n_valid = int(df_valid_tmp.count())
        mlflow.log_metrics({"n_df_model": n_model, "n_df_validacao": n_valid})

        if DO_PROFILE:
            mlflow.log_dict(profile_basic(df_model_tmp, "df_model", [ID_COL, SEG_COL]), "profiling_df_model.json")
            mlflow.log_dict(profile_basic(df_valid_tmp, "df_validacao", [ID_COL, SEG_COL]), "profiling_df_validacao.json")
            mlflow.log_dict({
                "counts_by_seg":     counts_by_seg(df_model_tmp, SEG_COL),
                "label_rate_by_seg": label_rate_by_seg(df_model_tmp, SEG_COL, LABEL_COL),
            }, "eda_df_model_by_seg.json")

        df_model_tmp.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_MODEL_FQN)
        df_valid_tmp.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_VALID_FQN)
        mlflow.log_metrics({"df_model_saved": 1, "df_validacao_saved": 1})

print("✅ T_PRE_PROC_MODEL ok")
print("• df_model    :", DF_MODEL_FQN, f"({n_model} linhas)")
print("• df_validacao:", DF_VALID_FQN, f"({n_valid} linhas)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_FEATURE_SELECTION

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports e helpers — FS

# COMMAND ----------

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from sklearn.feature_selection import mutual_info_classif


def truncate_high_cardinality(df: DataFrame, col_name: str, top_n: int, outros: str = "OUTROS") -> DataFrame:
    """
    Mantém as top_n categorias mais frequentes; demais (incluindo nulos/brancos) → outros.
    Frequências calculadas sem dependência do label — sem leakage.
    """
    top_vals = (
        df.filter(F.col(col_name).isNotNull() & (F.trim(F.col(col_name)) != ""))
          .groupBy(col_name).count()
          .orderBy(F.col("count").desc())
          .limit(top_n)
          .select(col_name)
          .rdd.flatMap(lambda x: x)
          .collect()
    )
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(top_vals), F.col(col_name)).otherwise(F.lit(outros)),
    )


def compute_ap(df_pred: DataFrame, label_col: str = "label", prob_col: str = "probability") -> float:
    """Average Precision: média das precisões nas posições onde label=1, ordenado por score desc."""
    p1    = vector_to_array(F.col(prob_col)).getItem(1).cast("double")
    n_pos = int(df_pred.filter(F.col(label_col) == 1).count())
    if n_pos == 0:
        return 0.0
    w_desc = Window.orderBy(F.col("_p1").desc())
    w_cum  = w_desc.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df_ap  = (
        df_pred
        .select(F.col(label_col).cast("int").alias("_y"), p1.alias("_p1"))
        .withColumn("_rn",     F.row_number().over(w_desc))
        .withColumn("_tp_cum", F.sum("_y").over(w_cum))
        .withColumn("_prec_i", F.col("_tp_cum") / F.col("_rn"))
    )
    ap = df_ap.filter(F.col("_y") == 1).agg(F.avg("_prec_i").alias("ap")).collect()[0]["ap"]
    return float(ap) if ap is not None else 0.0


def get_vector_attr_names(df_vec: DataFrame, vec_col: str = "features_vec") -> list:
    meta  = df_vec.schema[vec_col].metadata.get("ml_attr", {})
    attrs = []
    for k in ["binary", "numeric", "nominal"]:
        attrs.extend(meta.get("attrs", {}).get(k, []))
    return [a["name"] for a in sorted(attrs, key=lambda x: x["idx"])]


def base_feature_from_attr(attr_name: str) -> str:
    for sep in ["=", "__ohe", "__imp"]:
        if sep in attr_name:
            return attr_name.split(sep, 1)[0]
    return attr_name


def importance_to_score(model_key: str, importances: list, attr_names: list, all_features: list) -> pd.DataFrame:
    """
    Agrega importâncias de atributos OHE/IMP pela feature original.
    Normaliza para score [0,1] baseado em rank. Garante presença de todas as features (fillna 0).
    """
    pdf_raw = pd.DataFrame({"attr": attr_names, "importance": importances})
    pdf_raw["feature"] = pdf_raw["attr"].apply(base_feature_from_attr)

    pdf_agg = (
        pdf_raw.groupby("feature", as_index=False)["importance"]
               .sum()
               .sort_values("importance", ascending=False)
               .reset_index(drop=True)
    )
    n = len(pdf_agg)
    if n <= 1:
        pdf_agg["score"] = 1.0
    else:
        pdf_agg["rank_raw"] = pdf_agg["importance"].rank(method="first", ascending=False).astype(int)
        pdf_agg["score"]    = 1.0 - (pdf_agg["rank_raw"] - 1) / (n - 1)

    pdf_all = pd.DataFrame({"feature": all_features})
    pdf_all = pdf_all.merge(pdf_agg[["feature", "importance", "score"]], on="feature", how="left")
    pdf_all["importance"] = pdf_all["importance"].fillna(0.0)
    pdf_all["score"]      = pdf_all["score"].fillna(0.0)

    return pdf_all.rename(columns={
        "importance": f"importance_{model_key}",
        "score":      f"score_{model_key}",
    })


def log_pandas_csv(pdf: pd.DataFrame, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        fp   = os.path.join(td, os.path.basename(artifact_path))
        dir_ = os.path.dirname(artifact_path) if "/" in artifact_path else None
        pdf.to_csv(fp, index=False)
        mlflow.log_artifact(fp, artifact_path=dir_)


print("✅ Helpers FS carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução — T_FEATURE_SELECTION

# COMMAND ----------

RUN_FS_EXEC = run_name_vts("T_FS")

_fs_kw = {"run_id": FS_RUN_ID_OVERRIDE} if FS_RUN_ID_OVERRIDE else {"run_name": STEP_FEATURE_SELECTION_NAME}

with mlflow.start_run(**_fs_kw, nested=True) as fs_container:
    FS_CONTAINER_RUN_ID = fs_container.info.run_id

    with mlflow.start_run(run_name=RUN_FS_EXEC, nested=True):

        # ── Tags ──────────────────────────────────────────────────────────
        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "FEATURE_SELECTION",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })

        # ── Params: decisões de design rastreáveis ────────────────────────
        mlflow.log_params({
            "df_model_fqn":            DF_MODEL_FQN,
            "seg_target":              SEG_TARGET,
            "fs_seeds":                json.dumps(FS_SEEDS),
            "fs_train_frac":           FS_TRAIN_FRAC,
            "null_drop_pct":           NULL_DROP_PCT,
            "high_card_threshold":     HIGH_CARD_THRESHOLD,
            "high_card_top_n":         HIGH_CARD_TOP_N,
            "outros_label":            OUTROS_LABEL,
            "fs_methods":              json.dumps(list(FS_METHODS_CONFIG.keys())),
            "fs_methods_config":       json.dumps(FS_METHODS_CONFIG),
            "topk_list":               json.dumps(TOPK_LIST),
            "ensemble_type":           "weighted_auc_pr",
            "mi_parallel":             "true",
            "mi_in_ensemble":          "false",
            "pearson_exploratorio":    "true",
            "run_suffix":              RUN_SUFFIX,
            "ts_exec":                 TS_EXEC,
            "mode_code":               MODE_CODE,
            "pr_run_id":               PR_RUN_ID,
            "mode_run_id":             MODE_RUN_ID,
            "fs_container_run_id":     FS_CONTAINER_RUN_ID,
        })
        mlflow.log_dict(
            rules_catalog_for_logging({"rules_feature_prep": RULES_FEATURE_PREP}),
            "rules_feature_prep_catalog.json",
        )
        mlflow.log_dict(build_tables_lineage_fs(), "tables_lineage.json")

        # ── [1] Load + filter SEG_TARGET ──────────────────────────────────
        df_raw = spark.table(DF_MODEL_FQN)
        df_seg = df_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()
        n_rows_seg = int(df_seg.count())
        mlflow.log_metric("n_rows_seg", n_rows_seg)
        if n_rows_seg == 0:
            raise ValueError(f"❌ Nenhuma linha para SEG_TARGET='{SEG_TARGET}' em {DF_MODEL_FQN}")

        # ── Derivar cat/num a partir do schema e dos toggles FEATURE_CANDIDATES ──
        _schema_fs   = dict(df_seg.dtypes)
        _fc_enabled  = [c for c, v in FEATURE_CANDIDATES.items() if v     and c in _schema_fs]
        _fc_disabled = [c for c, v in FEATURE_CANDIDATES.items() if not v and c in _schema_fs]
        _fc_absent   = [c for c in FEATURE_CANDIDATES if c not in _schema_fs]

        FS_CAT_COLS = [c for c in _fc_enabled if _schema_fs[c] == "string"]
        FS_NUM_COLS = [c for c in _fc_enabled if _schema_fs[c] != "string"]
        # Aliases mantidos para compatibilidade downstream (DECIMAL e DIAS unificados em NUM)
        FS_DECIMAL_COLS     = FS_NUM_COLS
        FS_DIAS_COLS        = []
        EXCLUIR_DE_FEATURES = set()

        if _fc_absent:
            print(f"⚠️  Colunas em FEATURE_CANDIDATES ausentes na tabela: {_fc_absent}")
        print(f"• cat_cols ({len(FS_CAT_COLS)}): {FS_CAT_COLS}")
        print(f"• num_cols ({len(FS_NUM_COLS)}): {FS_NUM_COLS}")
        print(f"• disabled ({len(_fc_disabled)}): {_fc_disabled}")

        mlflow.log_params({
            "feature_candidates_enabled":  json.dumps(_fc_enabled),
            "feature_candidates_disabled": json.dumps(_fc_disabled),
            "feature_type_cat":            json.dumps(FS_CAT_COLS),
            "feature_type_num":            json.dumps(FS_NUM_COLS),
        })

        # ── [2] Cleaning ──────────────────────────────────────────────────
        df_clean = df_seg
        for c, t in df_clean.dtypes:
            if t == "string":
                df_clean = df_clean.withColumn(
                    c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c))
                )

        df_lab = df_clean.withColumn("label_int", F.col(LABEL_COL).cast("int"))
        n_label_invalid = int(
            df_lab.filter(~F.col("label_int").isin([0, 1]) | F.col("label_int").isNull()).count()
        )
        mlflow.log_metric("n_label_invalid_or_null", n_label_invalid)

        df_base = df_lab.drop(LABEL_COL).withColumnRenamed("label_int", "label")
        for c in [c for c in FS_DECIMAL_COLS + FS_DIAS_COLS if c in df_base.columns]:
            df_base = df_base.withColumn(c, F.col(c).cast("double"))

        # ── [3] Perfil de nulos — informativo (PP_R04 já aplicado em T_PRE_PROC_MODEL) ──
        cols_candidate_raw = FS_DECIMAL_COLS + FS_DIAS_COLS + FS_CAT_COLS
        cols_candidate     = [c for c in cols_candidate_raw if c in df_base.columns and c not in EXCLUIR_DE_FEATURES]
        null_counts        = df_base.agg(*[
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in cols_candidate
        ]).collect()[0].asDict()
        null_profile   = [{"col": c, "nulls": int(null_counts[c]), "pct_null": null_counts[c] / n_rows_seg}
                          for c in cols_candidate]
        # PP_R04 já aplicado em T_PRE_PROC_MODEL — colunas null-heavy removidas do df
        drop_null_cols = set()
        mlflow.log_dict(
            {"null_profile": null_profile, "drop_null_cols": [],
             "note": "PP_R04 aplicado em T_PRE_PROC_MODEL — colunas null-heavy já removidas do df"},
            "fs_stage1/null_profile.json",
        )

        # ── [4] Cardinalidade — informativo (PP_R05 já aplicado em T_PRE_PROC_MODEL) ──
        cat_present  = [c for c in FS_CAT_COLS if c in df_base.columns and c not in EXCLUIR_DE_FEATURES]
        cat_card_pre = [{"col": c, "count_distinct": int(df_base.select(F.countDistinct(c)).collect()[0][0])}
                        for c in cat_present]
        high_card_cols = [r["col"] for r in cat_card_pre if r["count_distinct"] > HIGH_CARD_THRESHOLD]
        # PP_R05 já aplicado em T_PRE_PROC_MODEL — sem truncagem nesta etapa
        mlflow.log_dict(
            {
                "cat_cardinality": cat_card_pre,
                "high_card_cols":  high_card_cols,
                "note":            "PP_R05 aplicado em T_PRE_PROC_MODEL — truncagem já realizada no df",
            },
            "fs_stage1/cat_cardinality.json",
        )

        # ── [5] Listas finais de features ─────────────────────────────────
        _id_cols_fs     = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
        _drop_base_fs   = _id_cols_fs + [STATUS_COL]
        # PP_R06 já aplicado em T_PRE_PROC_MODEL — colunas constantes removidas do df
        drop_constant_cols  = set()
        DROP_FEATURES_FINAL = set(_drop_base_fs) | drop_null_cols | drop_constant_cols | EXCLUIR_DE_FEATURES

        NUM_COLS_FINAL   = [c for c in FS_DECIMAL_COLS + FS_DIAS_COLS
                            if c in df_base.columns and c not in DROP_FEATURES_FINAL]
        CAT_COLS_FINAL   = [c for c in FS_CAT_COLS
                            if c in df_base.columns and c not in DROP_FEATURES_FINAL]
        FEATURE_COLS_ALL = CAT_COLS_FINAL + NUM_COLS_FINAL

        if not FEATURE_COLS_ALL:
            raise ValueError("❌ FEATURE_COLS_ALL vazio após drops.")

        mlflow.log_dict({
            "excluir_de_features": sorted(EXCLUIR_DE_FEATURES),
            "drop_features_final": sorted(DROP_FEATURES_FINAL),
            "cat_cols_final":      CAT_COLS_FINAL,
            "num_cols_final":      NUM_COLS_FINAL,
            "features_total":      len(FEATURE_COLS_ALL),
            "note":                "PP_R04/PP_R05/PP_R06 aplicados em T_PRE_PROC_MODEL",
        }, "fs_stage1/fs_feature_contract.json")
        mlflow.log_metric("n_features_candidate", len(FEATURE_COLS_ALL))

        df_audit = df_base.select(ID_COL, "label", *FEATURE_COLS_ALL).cache()

        # ── [6] Loop multi-seed: LR-L1, RF, GBT ──────────────────────────
        method_seed_results: dict = {m: [] for m in FS_METHODS_CONFIG}
        evaluator_pr = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
        )

        for seed in FS_SEEDS:
            fractions = {0: FS_TRAIN_FRAC, 1: FS_TRAIN_FRAC}
            df_tr     = df_audit.sampleBy("label", fractions=fractions, seed=seed).cache()
            tr_ids    = df_tr.select(ID_COL).cache()
            df_va     = df_audit.join(tr_ids, on=ID_COL, how="left_anti").cache()

            mlflow.log_metrics({
                f"n_train_seed{seed}": int(df_tr.count()),
                f"n_val_seed{seed}":   int(df_va.count()),
            })

            df_tr_ml = df_tr.select("label", *FEATURE_COLS_ALL)
            df_va_ml = df_va.select("label", *FEATURE_COLS_ALL)

            # pipeline fit apenas no train
            idx_cols  = [f"{c}__idx" for c in CAT_COLS_FINAL]
            ohe_cols  = [f"{c}__ohe" for c in CAT_COLS_FINAL]
            imp_cols  = [f"{c}__imp" for c in NUM_COLS_FINAL]
            pp_stages = []
            for c, out_c in zip(CAT_COLS_FINAL, idx_cols):
                pp_stages.append(StringIndexer(inputCol=c, outputCol=out_c, handleInvalid="keep"))
            if idx_cols:
                pp_stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, dropLast=False))
            if NUM_COLS_FINAL:
                pp_stages.append(Imputer(inputCols=NUM_COLS_FINAL, outputCols=imp_cols, strategy="mean"))
            pp_stages.append(VectorAssembler(inputCols=ohe_cols + imp_cols, outputCol="features_vec"))

            preprocess = Pipeline(stages=pp_stages).fit(df_tr_ml)
            df_tr_vec  = preprocess.transform(df_tr_ml).select("label", "features_vec").cache()
            df_va_vec  = preprocess.transform(df_va_ml).select("label", "features_vec").cache()
            attr_names = get_vector_attr_names(df_tr_vec)

            for method, cfg in FS_METHODS_CONFIG.items():
                if method == "lr_l1":
                    clf = LogisticRegression(
                        featuresCol="features_vec", labelCol="label", maxIter=cfg["maxIter"],
                        regParam=cfg["regParam"], elasticNetParam=cfg["elasticNetParam"], standardization=True,
                    )
                    clf_model   = clf.fit(df_tr_vec)
                    importances = [abs(float(x)) for x in clf_model.coefficients.toArray()]
                elif method == "rf":
                    clf = RandomForestClassifier(
                        featuresCol="features_vec", labelCol="label",
                        numTrees=cfg["numTrees"], maxDepth=cfg["maxDepth"], seed=seed,
                    )
                    clf_model   = clf.fit(df_tr_vec)
                    importances = [float(x) for x in clf_model.featureImportances.toArray()]
                elif method == "gbt":
                    clf = GBTClassifier(
                        featuresCol="features_vec", labelCol="label",
                        maxIter=cfg["maxIter"], maxDepth=cfg["maxDepth"], stepSize=cfg["stepSize"], seed=seed,
                    )
                    clf_model   = clf.fit(df_tr_vec)
                    importances = [float(x) for x in clf_model.featureImportances.toArray()]
                else:
                    raise ValueError(f"Método não suportado: {method}")

                pred_va    = clf_model.transform(df_va_vec)
                ap_val     = compute_ap(pred_va)
                auc_pr_val = float(evaluator_pr.evaluate(pred_va))

                mlflow.log_metrics({
                    f"{method}_seed{seed}_ap_val":     ap_val,
                    f"{method}_seed{seed}_auc_pr_val": auc_pr_val,
                })

                pdf_scored = importance_to_score(method, importances, attr_names, FEATURE_COLS_ALL)
                method_seed_results[method].append({
                    "seed": seed, "ap": ap_val, "auc_pr": auc_pr_val, "pdf_scored": pdf_scored,
                })
                log_pandas_csv(
                    pdf_scored.sort_values(f"score_{method}", ascending=False),
                    f"methods/{method}/seed{seed}/importance_by_feature.csv",
                )

            df_tr.unpersist(); df_va.unpersist()
            df_tr_vec.unpersist(); df_va_vec.unpersist()

        # ── [7] Agregação por método + ensemble ponderado ─────────────────
        method_avg: dict = {}

        for method, results in method_seed_results.items():
            avg_ap     = float(np.mean([r["ap"]     for r in results]))
            avg_auc_pr = float(np.mean([r["auc_pr"] for r in results]))
            score_col  = f"score_{method}"
            imp_col    = f"importance_{method}"

            pdf_avg = pd.DataFrame({
                "feature": FEATURE_COLS_ALL,
                score_col: np.array([r["pdf_scored"][score_col].values for r in results]).mean(axis=0),
                imp_col:   np.array([r["pdf_scored"][imp_col].values   for r in results]).mean(axis=0),
            })
            method_avg[method] = {"avg_ap": avg_ap, "avg_auc_pr": avg_auc_pr, "pdf_avg": pdf_avg}

            mlflow.log_metrics({
                f"{method}_avg_ap_val":     avg_ap,
                f"{method}_avg_auc_pr_val": avg_auc_pr,
            })
            log_pandas_csv(pdf_avg.sort_values(score_col, ascending=False),
                           f"methods/{method}/importance_avg.csv")

        total_w          = sum(v["avg_auc_pr"] for v in method_avg.values()) or 1.0
        ensemble_weights = {m: float(method_avg[m]["avg_auc_pr"] / total_w) for m in method_avg}

        pdf_final = pd.DataFrame({"feature": FEATURE_COLS_ALL})
        for method, info in method_avg.items():
            score_col = f"score_{method}"
            pdf_final = pdf_final.merge(info["pdf_avg"][["feature", score_col]], on="feature", how="left")
            pdf_final[f"weight_{method}"] = ensemble_weights[method]

        pdf_final["score_final"] = sum(
            pdf_final[f"score_{m}"].fillna(0.0) * pdf_final[f"weight_{m}"]
            for m in method_avg
        )
        pdf_final = pdf_final.sort_values("score_final", ascending=False).reset_index(drop=True)
        pdf_final["rank_final"] = range(1, len(pdf_final) + 1)

        features_ranked = pdf_final["feature"].tolist()
        feature_sets    = {f"top_{k}": features_ranked[:k] for k in TOPK_LIST}

        mlflow.log_dict(ensemble_weights, "summary/ensemble_weights.json")
        log_pandas_csv(pdf_final, "summary/feature_ranking_final.csv")
        mlflow.log_dict({"features_ranked": features_ranked}, "summary/features_ranked.json")
        mlflow.log_dict(feature_sets, "summary/topk_sets.json")
        mlflow.log_dict({
            "methods": list(method_avg.keys()), "seeds": FS_SEEDS,
            "ensemble_weights": ensemble_weights,
            "method_metrics_avg": {
                m: {"avg_ap": v["avg_ap"], "avg_auc_pr": v["avg_auc_pr"]}
                for m, v in method_avg.items()
            },
        }, "summary/methods_summary.json")

        # ── [8] Mutual Information (paralelo — não entra no ensemble) ─────
        MI_SAMPLE_SIZE = 50_000
        pdf_mi         = df_audit.select("label", *FEATURE_COLS_ALL).limit(MI_SAMPLE_SIZE).toPandas()
        pdf_mi_enc     = pdf_mi.copy()

        for c in CAT_COLS_FINAL:
            if c in pdf_mi_enc.columns:
                pdf_mi_enc[c] = pdf_mi_enc[c].astype("category").cat.codes.replace(-1, np.nan)
        for c in pdf_mi_enc.columns:
            if c != "label":
                med = pdf_mi_enc[c].median()
                pdf_mi_enc[c] = pdf_mi_enc[c].fillna(med if pd.notna(med) else 0.0)

        X_mi          = pdf_mi_enc[FEATURE_COLS_ALL].values.astype(float)
        y_mi          = pdf_mi_enc["label"].values.astype(int)
        mi_scores_raw = mutual_info_classif(X_mi, y_mi, random_state=42)

        pdf_mi_result = (
            pd.DataFrame({"feature": FEATURE_COLS_ALL, "mi_score": mi_scores_raw})
              .sort_values("mi_score", ascending=False)
              .reset_index(drop=True)
        )
        pdf_mi_result["mi_rank"] = range(1, len(pdf_mi_result) + 1)

        log_pandas_csv(pdf_mi_result, "mi/mutual_information.csv")
        mlflow.log_dict({
            "mi_scores":   pdf_mi_result.to_dict(orient="records"),
            "sample_size": MI_SAMPLE_SIZE,
            "note":        "MI paralelo — nao entra no ensemble ponderado",
        }, "mi/mutual_information.json")

        # ── [9] Correlação de Pearson (nova — MODE_C) ────────────────────
        # Exploratória sobre features numéricas do ranking final.
        # Não influencia a seleção — apenas registra correlações para análise.
        if len(NUM_COLS_FINAL) >= 2:
            PEARSON_SAMPLE_SIZE = 50_000
            pdf_pearson = df_audit.select(*NUM_COLS_FINAL).limit(PEARSON_SAMPLE_SIZE).toPandas()
            corr_matrix = pdf_pearson[NUM_COLS_FINAL].corr(method="pearson")

            # Formato longo
            corr_long         = corr_matrix.stack().reset_index()
            corr_long.columns = ["feature_a", "feature_b", "correlation"]
            log_pandas_csv(corr_long, "pearson/pearson_correlation.csv")

            # Heatmap
            n_num = len(NUM_COLS_FINAL)
            fig_size = max(6, n_num)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(n_num))
            ax.set_yticks(range(n_num))
            ax.set_xticklabels(NUM_COLS_FINAL, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(NUM_COLS_FINAL, fontsize=8)
            for i in range(n_num):
                for j in range(n_num):
                    ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                            ha="center", va="center", fontsize=7)
            ax.set_title(f"Pearson Correlation — features numéricas (sample={PEARSON_SAMPLE_SIZE})")
            plt.tight_layout()

            with tempfile.TemporaryDirectory() as td:
                fp = os.path.join(td, "pearson_heatmap.png")
                fig.savefig(fp, dpi=120, bbox_inches="tight")
                mlflow.log_artifact(fp, artifact_path="pearson")
            plt.close(fig)

            mlflow.log_dict({
                "num_cols":    NUM_COLS_FINAL,
                "sample_size": PEARSON_SAMPLE_SIZE,
                "note":        "Pearson exploratorio — nao entra no ensemble",
            }, "pearson/pearson_config.json")
        else:
            print("ℹ️  Pearson skipped: menos de 2 colunas numéricas finais")

        # ── Variáveis para o step TREINO ──────────────────────────────────
        FS_FEATURES_RANKED = features_ranked
        FS_FEATURE_SETS    = feature_sets
        FS_CAT_COLS_FINAL  = CAT_COLS_FINAL
        FS_NUM_COLS_FINAL  = NUM_COLS_FINAL

        print("✅ T_FEATURE_SELECTION ok")
        print("• run         :", RUN_FS_EXEC)
        print("• features    :", len(FEATURE_COLS_ALL), "candidatas")
        print("• weights     :", {m: f"{w:.3f}" for m, w in ensemble_weights.items()})
        print("• top_5       :", FS_FEATURE_SETS.get("top_5"))
        print("• top_7       :", FS_FEATURE_SETS.get("top_7"))
        print("• top_12      :", FS_FEATURE_SETS.get("top_12"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_TREINO

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config — T_TREINO
# MAGIC
# MAGIC Ajustar `TREINO_FEATURE_SET_KEY` com base nos resultados do FS acima.
# MAGIC Não há `CAPACIDADE_PCT` nem `LIFT_TARGET` — avaliação de capacidade delegada ao 5_COMP_MODE_C.

# COMMAND ----------

# Chave do feature set gerado pelo FS — deve existir em FS_FEATURE_SETS
TREINO_FEATURE_SET_KEY = "top_5"  # <<< AJUSTE

# Class weight
USE_CLASS_WEIGHT       = "auto"   # "auto" | True | False
CLASS_WEIGHT_THRESHOLD = 0.30

# CV + grid
CV_FOLDS  = 5
CV_SEED   = 42
CV_METRIC = "areaUnderPR"

# Grid de hiperparâmetros
# Listas: variam no grid → cada combinação gera um model_id.
# Scalar (maxIter): fixo em todos os combos.
# Exemplo com 1 combo: maxDepth=[4], stepSize=[0.1] → d4_s01
# Exemplo com 4 combos: maxDepth=[4,6], stepSize=[0.1,0.05] → d4_s01, d4_s005, d6_s01, d6_s005
GBT_PARAM_GRID = {
    "maxDepth": [4, 6],       # <<< adicionar valores para mais combos, ex: [4, 6]
    "stepSize": [0.05, 0.1],     # <<< adicionar valores para mais combos, ex: [0.1, 0.05]
    "maxIter":  100,
}

# Eval — critério de seleção de τ na curva PR
# "max_f1"              → τ que maximiza F1 (equilíbrio precision/recall)
# "max_f2"              → τ que maximiza F2 (peso duplo no recall)
# "precision_ge_target" → maior recall com precision >= EVAL_PRECISION_TARGET
EVAL_CRITERION        = "max_f1"
EVAL_PRECISION_TARGET = 0.25   # usado apenas quando EVAL_CRITERION = "precision_ge_target"

ID_COLS            = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]

TREINO_FEATURE_COLS = FS_FEATURE_SETS[TREINO_FEATURE_SET_KEY]

print("✅ T_TREINO inputs:")
print("• df_model_fqn :", DF_MODEL_FQN)
print("• df_valid_fqn :", DF_VALID_FQN)
print("• feature_set  :", TREINO_FEATURE_SET_KEY, "→", TREINO_FEATURE_COLS)
print("• use_cw       :", USE_CLASS_WEIGHT, "| threshold:", CLASS_WEIGHT_THRESHOLD)
print("• cv_folds     :", CV_FOLDS, "| cv_seed:", CV_SEED)
print("• param_grid   :", GBT_PARAM_GRID)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports e helpers — T_TREINO

# COMMAND ----------

import itertools

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def build_preprocess_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    idx_cols = [f"{c}__idx" for c in cat_cols]
    ohe_cols = [f"{c}__ohe" for c in cat_cols]
    imp_cols = [f"{c}__imp" for c in num_cols]
    stages = []
    for c, out_c in zip(cat_cols, idx_cols):
        stages.append(StringIndexer(inputCol=c, outputCol=out_c, handleInvalid="keep"))
    if idx_cols:
        stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, dropLast=False))
    if num_cols:
        stages.append(Imputer(inputCols=num_cols, outputCols=imp_cols, strategy="mean"))
    stages.append(VectorAssembler(inputCols=ohe_cols + imp_cols, outputCol="features_vec"))
    return Pipeline(stages=stages)


def add_class_weights(df: DataFrame, label_col: str = "label",
                      use_class_weight="auto", class_weight_threshold: float = 0.30) -> tuple:
    """Retorna (df_com_weight, label_rate, weight_pos, apply_cw)."""
    label_rate = float(df.agg(F.avg(F.col(label_col).cast("double"))).collect()[0][0])
    apply_cw = (
        use_class_weight is True
        or (use_class_weight == "auto" and label_rate < class_weight_threshold)
    )
    if apply_cw:
        weight_pos = (1.0 - label_rate) / label_rate
        weight_neg = 1.0
        df = df.withColumn(
            "weight",
            F.when(F.col(label_col) == 1.0, F.lit(weight_pos)).otherwise(F.lit(weight_neg)),
        )
    else:
        weight_pos = 1.0
        df = df.withColumn("weight", F.lit(1.0))
    return df, label_rate, weight_pos, apply_cw


def kfold_split(df: DataFrame, n_folds: int, seed: int, id_col: str):
    """Divide df em n_folds folds via hash determinístico. Retorna lista de (df_train, df_val)."""
    df_h = df.withColumn(
        "_fold",
        F.pmod(F.abs(F.xxhash64(F.col(id_col).cast("string"), F.lit(seed))), F.lit(n_folds)).cast("int"),
    )
    folds = []
    for i in range(n_folds):
        df_val   = df_h.filter(F.col("_fold") == i).drop("_fold")
        df_train = df_h.filter(F.col("_fold") != i).drop("_fold")
        folds.append((df_train, df_val))
    return folds


def build_tables_lineage_treino() -> dict:
    return {
        "stage":         "T_TREINO",
        "ts_exec":       TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode":          MODE_CODE,
        "inputs":        {"df_model": DF_MODEL_FQN, "df_validacao": DF_VALID_FQN},
        "seg_target":    SEG_TARGET,
        "feature_set":   TREINO_FEATURE_SET_KEY,
    }


print("✅ Helpers T_TREINO carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helpers — T_EVAL

# COMMAND ----------

def compute_pr_curve(pdf: pd.DataFrame, label_col: str = "label", score_col: str = "p1") -> pd.DataFrame:
    """
    Varre thresholds de 0.01 a 0.99 e calcula precision, recall, F1 e F2 para cada ponto.
    Retorna DataFrame pandas com colunas: threshold, tp, fp, fn, tn, precision, recall, f1, f2.
    """
    y     = pdf[label_col].values
    score = pdf[score_col].values
    rows  = []
    for t in [round(v / 100, 2) for v in range(1, 100)]:
        yhat      = (score >= t).astype(int)
        tp        = int(((y == 1) & (yhat == 1)).sum())
        fp        = int(((y == 0) & (yhat == 1)).sum())
        fn        = int(((y == 1) & (yhat == 0)).sum())
        tn        = int(((y == 0) & (yhat == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall    = tp / (tp + fn) if (tp + fn) > 0 else None
        f1        = (2 * precision * recall / (precision + recall))     if (precision and recall) else None
        f2        = (5 * precision * recall / (4 * precision + recall)) if (precision and recall) else None
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": precision, "recall": recall, "f1": f1, "f2": f2})
    return pd.DataFrame(rows)


def select_threshold(pdf_curve: pd.DataFrame, criterion: str, precision_target: float = 0.25) -> dict:
    """Seleciona a linha da curva correspondente ao τ ótimo segundo o critério configurado."""
    df = pdf_curve.dropna(subset=["precision", "recall"])
    if criterion == "max_f1":
        row = df.loc[df["f1"].idxmax()]
    elif criterion == "max_f2":
        row = df.loc[df["f2"].idxmax()]
    elif criterion == "precision_ge_target":
        df_ok = df[df["precision"] >= precision_target]
        if df_ok.empty:
            raise ValueError(f"Nenhum threshold satisfaz precision >= {precision_target}")
        row = df_ok.loc[df_ok["recall"].idxmax()]
    else:
        raise ValueError(f"EVAL_CRITERION inválido: '{criterion}'. Use 'max_f1', 'max_f2' ou 'precision_ge_target'.")
    return row.to_dict()


def log_figure(fig, artifact_path: str) -> None:
    """Salva figura matplotlib e loga como artifact MLflow."""
    with tempfile.TemporaryDirectory() as td:
        fname  = os.path.basename(artifact_path)
        folder = os.path.dirname(artifact_path) if "/" in artifact_path else None
        fp     = os.path.join(td, fname)
        fig.savefig(fp, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(fp, artifact_path=folder)
    plt.close(fig)


def plot_pr_curve(pdf_curve: pd.DataFrame, tau_row: dict, baseline: float, model_id: str):
    """Curva PR com ponto τ marcado e linha de baseline (modelo aleatório)."""
    df  = pdf_curve.dropna(subset=["precision", "recall"]).sort_values("recall")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["recall"], df["precision"], color="steelblue", lw=2, label="Modelo")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.2,
               label=f"Baseline aleatório ({baseline:.3f})")
    ax.scatter([tau_row["recall"]], [tau_row["precision"]],
               color="crimson", zorder=5, s=80,
               label=f"τ={tau_row['threshold']:.2f}  P={tau_row['precision']:.3f}  R={tau_row['recall']:.3f}")
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.set_title(f"Curva PR — {model_id}")
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_threshold_metrics(pdf_curve: pd.DataFrame, tau_row: dict, model_id: str):
    """Precision, Recall e F1 em função de τ, com linha vertical no τ escolhido."""
    df  = pdf_curve.dropna(subset=["precision", "recall"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["precision"], label="Precision", color="steelblue",   lw=2)
    ax.plot(df["threshold"], df["recall"],    label="Recall",    color="darkorange",  lw=2)
    ax.plot(df["threshold"], df["f1"],        label="F1",        color="seagreen",    lw=2)
    ax.axvline(tau_row["threshold"], color="crimson", linestyle="--", lw=1.5,
               label=f"τ escolhido = {tau_row['threshold']:.2f}")
    ax.set_xlabel("Threshold (τ)");  ax.set_ylabel("Score")
    ax.set_title(f"Métricas por threshold — {model_id}")
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(tau_row: dict, model_id: str):
    """Heatmap da confusion matrix com valores absolutos."""
    cm     = np.array([[tau_row["tn"], tau_row["fp"]],
                       [tau_row["fn"], tau_row["tp"]]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1]);  ax.set_xticklabels(["Pred 0 (Perdida)", "Pred 1 (Emitida)"])
    ax.set_yticks([0, 1]);  ax.set_yticklabels(["Real 0 (Perdida)", "Real 1 (Emitida)"])
    ax.set_title(f"Confusion Matrix — {model_id}  (τ={tau_row['threshold']:.2f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


print("✅ Helpers T_EVAL carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução — T_TREINO

# COMMAND ----------

RUN_TREINO_EXEC = run_name_vts("T_TREINO")

_tr_kw = {"run_id": TREINO_RUN_ID_OVERRIDE} if TREINO_RUN_ID_OVERRIDE else {"run_name": STEP_TREINO_NAME}

with mlflow.start_run(**_tr_kw, nested=True) as treino_container:
    TREINO_CONTAINER_RUN_ID = treino_container.info.run_id

    with mlflow.start_run(run_name=RUN_TREINO_EXEC, nested=True) as treino_exec_run:
        TREINO_EXEC_RUN_ID = treino_exec_run.info.run_id

        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "TREINO",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })

        # ── [1] Carga, limpeza e truncagem de cardinalidade ───────────────
        df_model_raw = spark.table(DF_MODEL_FQN)
        df_model_seg = df_model_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))

        # Blank → null em strings
        for c, t in df_model_seg.dtypes:
            if t == "string":
                df_model_seg = df_model_seg.withColumn(
                    c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c))
                )

        # Identificar colunas do feature set por tipo
        treino_cat_cols = [c for c in TREINO_FEATURE_COLS if c in FS_CAT_COLS]
        treino_num_cols = [c for c in TREINO_FEATURE_COLS if c in FS_DECIMAL_COLS + FS_DIAS_COLS]

        # Cast numérico
        for c in treino_num_cols:
            df_model_seg = df_model_seg.withColumn(c, F.col(c).cast("double"))

        # Truncagem já aplicada em T_PRE_PROC_MODEL (PP_R05).
        # top_vals_by_col_prep calculado lá e disponível como variável Python.
        # Reutilizado aqui para salvar como artifact para o 4_INFERENCIA.
        top_vals_by_col = top_vals_by_col_prep

        # Class weight
        df_model_seg, label_rate, weight_pos, apply_cw = add_class_weights(
            df_model_seg, LABEL_COL, USE_CLASS_WEIGHT, CLASS_WEIGHT_THRESHOLD
        )
        n_model = int(df_model_seg.count())

        # Grid de combinações
        param_combinations = [
            {"maxDepth": d, "stepSize": s, "maxIter": GBT_PARAM_GRID["maxIter"]}
            for d, s in itertools.product(GBT_PARAM_GRID["maxDepth"], GBT_PARAM_GRID["stepSize"])
        ]
        model_id_list = [
            f"d{c['maxDepth']}_s{str(c['stepSize']).replace('.', '')}"
            for c in param_combinations
        ]

        mlflow.log_params({
            "df_model_fqn":            DF_MODEL_FQN,
            "df_valid_fqn":            DF_VALID_FQN,
            "seg_target":              SEG_TARGET,
            "feature_set":             TREINO_FEATURE_SET_KEY,
            "feature_cols":            json.dumps(TREINO_FEATURE_COLS),
            "n_features":              len(TREINO_FEATURE_COLS),
            "n_model":                 n_model,
            "use_class_weight":        str(USE_CLASS_WEIGHT),
            "class_weight_threshold":  CLASS_WEIGHT_THRESHOLD,
            "label_rate":              round(label_rate, 4),
            "apply_cw":                str(apply_cw),
            "weight_pos":              round(weight_pos, 4),
            "cv_folds":                CV_FOLDS,
            "cv_seed":                 CV_SEED,
            "cv_metric":               CV_METRIC,
            "gbt_param_grid":          json.dumps(GBT_PARAM_GRID),
            "gbt_maxiter_fixed":       GBT_PARAM_GRID["maxIter"],
            "model_ids":               json.dumps(model_id_list),
            "mode_code":               MODE_CODE,
            "pr_run_id":               PR_RUN_ID,
            "mode_run_id":             MODE_RUN_ID,
            "treino_container_run_id": TREINO_CONTAINER_RUN_ID,
            "treino_cat_cols":         json.dumps(treino_cat_cols),
            "treino_num_cols":         json.dumps(treino_num_cols),
            "note_mode_c":             "todos_combos_salvos_sem_selecao_vencedor",
        })

        mlflow.log_dict(build_tables_lineage_treino(), "tables_lineage.json")

        # Salvar top_vals_by_col como artifact (reutilizado pelo 4_INFERENCIA)
        mlflow.log_dict(top_vals_by_col, "preprocess/top_vals_by_col.json")

        df_model_ml = df_model_seg.select(ID_COL, LABEL_COL, "weight", *TREINO_FEATURE_COLS).cache()

        evaluator_pr = BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
        )

        # ── [2] CV 3-fold determinístico ──────────────────────────────────
        folds         = kfold_split(df_model_ml, CV_FOLDS, CV_SEED, ID_COL)
        grid_results  = []
        fold_metrics  = []

        for combo in param_combinations:
            combo_key = f"d{combo['maxDepth']}_s{str(combo['stepSize']).replace('.', '')}"
            fold_aucs = []

            for fold_i, (df_tr_fold, df_va_fold) in enumerate(folds):
                pp     = build_preprocess_pipeline(treino_cat_cols, treino_num_cols)
                pp_fit = pp.fit(df_tr_fold)
                df_tr_vec = pp_fit.transform(df_tr_fold).select(LABEL_COL, "weight", "features_vec").cache()
                df_va_vec = pp_fit.transform(df_va_fold).select(LABEL_COL, "features_vec").cache()

                gbt = GBTClassifier(
                    featuresCol="features_vec", labelCol=LABEL_COL,
                    weightCol="weight" if apply_cw else None,
                    maxDepth=combo["maxDepth"], stepSize=combo["stepSize"],
                    maxIter=combo["maxIter"], seed=CV_SEED,
                )
                gbt_fit  = gbt.fit(df_tr_vec)
                pred_val = gbt_fit.transform(df_va_vec)
                auc_pr   = float(evaluator_pr.evaluate(pred_val))
                fold_aucs.append(auc_pr)

                fold_metrics.append({
                    "combo": combo_key, "fold": fold_i,
                    "maxDepth": combo["maxDepth"], "stepSize": combo["stepSize"],
                    "auc_pr": round(auc_pr, 4),
                })

                mlflow.log_metric(f"cv_{combo_key}_fold{fold_i}_auc_pr", auc_pr)
                df_tr_vec.unpersist(); df_va_vec.unpersist()

            avg_auc_pr = float(np.mean(fold_aucs))
            std_auc_pr = float(np.std(fold_aucs))
            grid_results.append({
                "combo": combo_key, "maxDepth": combo["maxDepth"], "stepSize": combo["stepSize"],
                "avg_auc_pr": round(avg_auc_pr, 4), "std_auc_pr": round(std_auc_pr, 4),
                "fold_aucs": [round(x, 4) for x in fold_aucs],
            })
            mlflow.log_metrics({
                f"cv_{combo_key}_avg_auc_pr": avg_auc_pr,
                f"cv_{combo_key}_std_auc_pr": std_auc_pr,
            })

        mlflow.log_dict(grid_results, "cv/grid_results.json")
        mlflow.log_dict(fold_metrics, "cv/fold_metrics.json")

        print("✅ CV concluído")
        for r in sorted(grid_results, key=lambda x: x["avg_auc_pr"], reverse=True):
            print(f"  {r['combo']:20s} avg_auc_pr={r['avg_auc_pr']:.4f}  std={r['std_auc_pr']:.4f}")

        # ── [3] Treino final — TODOS os combos (sem seleção de vencedor) ──
        # Para cada combo: fit pipeline + fit GBT + save no MLflow + AUC-PR de treino.
        TRAINED_MODELS = {}

        for combo, cv_row in zip(param_combinations, grid_results):
            model_id = cv_row["combo"]
            print(f"\n  → Treinando final: {model_id}")

            # Pipeline fit em df_model completo
            pp_final     = build_preprocess_pipeline(treino_cat_cols, treino_num_cols)
            pp_final_fit = pp_final.fit(df_model_ml)
            df_model_vec = pp_final_fit.transform(df_model_ml).select(LABEL_COL, "weight", "features_vec").cache()

            gbt_final = GBTClassifier(
                featuresCol="features_vec", labelCol=LABEL_COL,
                weightCol="weight" if apply_cw else None,
                maxDepth=combo["maxDepth"], stepSize=combo["stepSize"],
                maxIter=combo["maxIter"], seed=CV_SEED,
            )
            gbt_model = gbt_final.fit(df_model_vec)

            # Salvar modelo e pipeline no artifact store do MLflow
            # Referência completa para o 4_INFERENCIA: (TREINO_EXEC_RUN_ID, model_id)
            mlflow.spark.log_model(gbt_model,   artifact_path=f"treino_final/{model_id}/model")
            mlflow.spark.log_model(pp_final_fit, artifact_path=f"treino_final/{model_id}/preprocess_pipeline")

            # AUC-PR e AP de treino (para análise de overfitting no 5_COMP)
            df_model_pred  = gbt_model.transform(df_model_vec)
            auc_pr_treino  = float(evaluator_pr.evaluate(df_model_pred))
            ap_treino      = compute_ap(df_model_pred, label_col=LABEL_COL)

            mlflow.log_metrics({
                f"auc_pr_treino_{model_id}": auc_pr_treino,
                f"ap_treino_{model_id}":     ap_treino,
            })

            TRAINED_MODELS[model_id] = {
                "model_id":        model_id,
                "params":          combo,
                "cv_avg_auc_pr":   cv_row["avg_auc_pr"],
                "cv_std_auc_pr":   cv_row["std_auc_pr"],
                "auc_pr_treino":   round(auc_pr_treino, 4),
                "ap_treino":       round(ap_treino, 4),
            }

            df_model_vec.unpersist()
            print(f"     auc_pr_treino={auc_pr_treino:.4f}  ap_treino={ap_treino:.4f}  cv_avg={cv_row['avg_auc_pr']:.4f}")

        # ── [4] Artifacts de resumo ────────────────────────────────────────
        mlflow.log_dict(list(TRAINED_MODELS.values()), "cv/trained_models_registry.json")
        df_model_ml.unpersist()

        # ── [5] T_EVAL — avaliação no conjunto de validação ───────────────
        df_eval = spark.table(DF_VALID_FQN).filter(F.col(SEG_COL) == F.lit(SEG_TARGET))
        for c, dtype in df_eval.dtypes:
            if dtype == "string":
                df_eval = df_eval.withColumn(
                    c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c))
                )
        for c in treino_num_cols:
            if c in df_eval.columns:
                df_eval = df_eval.withColumn(c, F.col(c).cast("double"))
        df_eval  = df_eval.cache()
        n_eval   = int(df_eval.count())
        baseline = float(df_eval.agg(F.avg(F.col(LABEL_COL).cast("double"))).collect()[0][0])

        eval_results    = {}
        eval_pdf_curves = {}
        eval_figs       = {}

        for model_id in model_id_list:
            print(f"\n  [EVAL] → {model_id}")

            pp_uri    = f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/preprocess_pipeline"
            model_uri = f"runs:/{TREINO_EXEC_RUN_ID}/treino_final/{model_id}/model"

            pp_loaded    = mlflow.spark.load_model(pp_uri)
            model_loaded = mlflow.spark.load_model(model_uri)

            df_vec  = pp_loaded.transform(df_eval.select(LABEL_COL, *TREINO_FEATURE_COLS))
            df_pred = model_loaded.transform(df_vec)

            p1_col     = vector_to_array(F.col("probability")).getItem(1).cast("double")
            pdf_scores = (df_pred
                          .select(F.col(LABEL_COL).cast("int").alias("label"), p1_col.alias("p1"))
                          .toPandas())

            pdf_curve = compute_pr_curve(pdf_scores)
            tau_row   = select_threshold(pdf_curve, EVAL_CRITERION, EVAL_PRECISION_TARGET)

            eval_results[model_id] = {
                "model_id":  model_id,
                "criterion": EVAL_CRITERION,
                "threshold": tau_row["threshold"],
                "precision": round(tau_row["precision"], 4),
                "recall":    round(tau_row["recall"],    4),
                "f1":        round(tau_row["f1"],        4),
                "f2":        round(tau_row["f2"],        4),
                "tp": tau_row["tp"], "fp": tau_row["fp"],
                "fn": tau_row["fn"], "tn": tau_row["tn"],
                "baseline": round(baseline, 4),
                "n_eval":   n_eval,
            }
            eval_pdf_curves[model_id] = pdf_curve
            eval_figs[model_id] = {
                "pr_curve":          plot_pr_curve(pdf_curve, tau_row, baseline, model_id),
                "threshold_metrics": plot_threshold_metrics(pdf_curve, tau_row, model_id),
                "confusion_matrix":  plot_confusion_matrix(tau_row, model_id),
            }

            print(f"     τ={tau_row['threshold']:.2f}  P={tau_row['precision']:.3f}  "
                  f"R={tau_row['recall']:.3f}  F1={tau_row['f1']:.3f}  F2={tau_row['f2']:.3f}")

        df_eval.unpersist()

        mlflow.log_param("eval_criterion",        EVAL_CRITERION)
        mlflow.log_param("eval_precision_target", EVAL_PRECISION_TARGET)
        mlflow.log_metric("eval_n_valid", float(n_eval))
        mlflow.log_metric("eval_baseline", round(baseline, 4))

        for model_id, r in eval_results.items():
            prefix = f"eval_{model_id}"
            mlflow.log_metrics({
                f"{prefix}_threshold": r["threshold"],
                f"{prefix}_precision": r["precision"],
                f"{prefix}_recall":    r["recall"],
                f"{prefix}_f1":        r["f1"],
                f"{prefix}_f2":        r["f2"],
                f"{prefix}_tp":        float(r["tp"]),
                f"{prefix}_fp":        float(r["fp"]),
                f"{prefix}_fn":        float(r["fn"]),
                f"{prefix}_tn":        float(r["tn"]),
            })
            log_pandas_csv(eval_pdf_curves[model_id], f"eval/{model_id}/pr_curve.csv")
            mlflow.log_dict(r, f"eval/{model_id}/eval_summary.json")
            log_figure(eval_figs[model_id]["pr_curve"],          f"eval/{model_id}/pr_curve.png")
            log_figure(eval_figs[model_id]["threshold_metrics"], f"eval/{model_id}/threshold_metrics.png")
            log_figure(eval_figs[model_id]["confusion_matrix"],  f"eval/{model_id}/confusion_matrix.png")

print("\n✅ T_TREINO + T_EVAL ok")
print("• TREINO_EXEC_RUN_ID :", TREINO_EXEC_RUN_ID)
print("• Modelos treinados  :", list(TRAINED_MODELS.keys()))
print("\nEVAL:")
for mid, r in eval_results.items():
    print(f"• {mid}: τ={r['threshold']}  P={r['precision']}  R={r['recall']}  F1={r['f1']}  F2={r['f2']}")
print("\n⚠️  Anotar TREINO_EXEC_RUN_ID para uso no 4_INFERENCIA_MODE_C e 5_COMP_MODE_C:")
print(f"    TREINO_EXEC_RUN_ID = \"{TREINO_EXEC_RUN_ID}\"")