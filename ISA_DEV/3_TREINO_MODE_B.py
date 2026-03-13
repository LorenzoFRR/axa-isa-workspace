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
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"

PR_TREINO_NAME  = "T_PR_TREINO"
MODE_CODE       = "B"
MODE_NAME       = f"T_MODE_{MODE_CODE}"


PR_RUN_ID_OVERRIDE        = "21f1184afb354159b325ad87bca3c50d"
MODE_RUN_ID_OVERRIDE      = ""
PRE_PROC_RUN_ID_OVERRIDE  = ""
FS_RUN_ID_OVERRIDE        = ""
TREINO_RUN_ID_OVERRIDE    = ""

STEP_PRE_PROC_NAME          = "T_PRE_PROC_MODEL"
STEP_FEATURE_SELECTION_NAME = "T_FEATURE_SELECTION"
STEP_TREINO_NAME            = "T_TREINO"

# =========================
# Versionamento
# =========================
TREINO_VERSAO            = "V8"
TREINO_VERSAO_TABLE_SAFE = TREINO_VERSAO.replace(".", "_")
VERSAO_REF               = TREINO_VERSAO

TS_EXEC  = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_UUID = uuid.uuid4().hex[:8]
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# =========================
# INPUT
# =========================
COTACAO_SEG_FQN = "silver.cotacao_seg_20260308_140026"  # <<< AJUSTE

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
SPLIT_SALT           = "split_b1_seg_mes"

DO_PROFILE = True

# =========================
# FS — params
# =========================
SEG_TARGET = "SEGURO_NOVO_MANUAL"  # <<< AJUSTE

FS_SEEDS            = [42, 123, 7]
FS_TRAIN_FRAC       = 0.70
NULL_DROP_PCT       = 0.90
HIGH_CARD_THRESHOLD = 15
HIGH_CARD_TOP_N     = 10
OUTROS_LABEL        = "OUTROS"

# Features candidatas — sem QTD_* e HR_* (leakage pendente de verificação)
FS_DECIMAL_COLS = [
    "VL_PREMIO_ALVO", "VL_PREMIO_LIQUIDO", "VL_PRE_TOTAL",
    "VL_ENDOSSO_PREMIO_TOTAL", "VL_GWP_CORRETOR_RESUMO",
]
FS_DIAS_COLS = [
    "DIAS_INICIO_VIGENCIA", "DIAS_VALIDADE", "DIAS_ANALISE_SUBSCRICAO",
    "DIAS_FIM_ANALISE_SUBSCRICAO", "DIAS_COTACAO", "DIAS_ULTIMA_ATUALIZACAO",
]
FS_CAT_COLS = [
    "INTERMENDIARIO_PERFIL",
    "DS_PRODUTO_NOME",
    "DS_SISTEMA",
    "CD_FILIAL_RESPONSAVEL_COTACAO",
    "DS_ATIVIDADE_SEGURADO",
    "DS_GRUPO_CORRETOR_SEGMENTO",
]

FS_METHODS_CONFIG = {
    "lr_l1": {"maxIter": 100, "regParam": 0.01, "elasticNetParam": 1.0},
    "rf":    {"numTrees": 200, "maxDepth": 8},
    "gbt":   {"maxIter": 80,  "maxDepth": 5, "stepSize": 0.1},
}

TOPK_LIST = [5, 7, 12]

# =========================
# TREINO — params
# =========================
# Capacidade do time: % do ranking em que o time consegue atuar.
# K = int(n_hold_out × CAPACIDADE_PCT) → define o corte do ranking.
CAPACIDADE_PCT = 0.10  # <<< AJUSTE após análise de capacidade operacional

# Lift mínimo desejado vs baseline.
# PRECISION_TARGET é derivado: LIFT_TARGET × baseline (não é um input direto).
LIFT_TARGET = 2.0  # <<< AJUSTE

# Baseline para cálculo de lift:
#   "taxa_base"      → label.mean() calculado no hold-out
#   "conversao_time" → valor informado em CONVERSAO_TIME (conversão atual do time)
BASELINE_MODE  = "conversao_time"  # <<< AJUSTE
CONVERSAO_TIME = 0.3         # float — preencher apenas se BASELINE_MODE="conversao_time"

ID_COLS            = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]

print("✅ CONFIG MODE_B carregada")
print("• input         :", COTACAO_SEG_FQN)
print("• mode          :", MODE_CODE)
print("• versao        :", TREINO_VERSAO)
print("• capacidade_pct:", CAPACIDADE_PCT)
print("• lift_target   :", LIFT_TARGET)
print("• baseline_mode :", BASELINE_MODE)
if BASELINE_MODE == "conversao_time":
    print("• conversao_time:", CONVERSAO_TIME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers globais

# COMMAND ----------

import json
from typing import Callable, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

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
    # MODE_B: sem conversão de status intermediários para Perdida.
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

RULES_BY_BLOCK = {
    "rules_on_df_seg":          RULES_ON_DF_SEG,
    "rules_build_base":         RULES_BUILD_BASE,
    "rules_build_df_model":     RULES_BUILD_MODEL,
    "rules_build_df_validacao": RULES_BUILD_VALID,
}

print("✅ RULES_BY_BLOCK definido")
print("• rules_on_df_seg         :", len(RULES_ON_DF_SEG))
print("• rules_build_base        :", len(RULES_BUILD_BASE))
print("• rules_build_df_model    :", len(RULES_BUILD_MODEL))
print("• rules_build_df_validacao:", len(RULES_BUILD_VALID))

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
            "input_cotacao_seg_fqn": COTACAO_SEG_FQN,
            "df_model_fqn": DF_MODEL_FQN, "df_validacao_fqn": DF_VALID_FQN,
            "valid_frac": float(VALID_FRAC), "split_salt": SPLIT_SALT,
            "allowed_final_status": json.dumps(ALLOWED_FINAL_STATUS),
            "label_col": LABEL_COL, "id_col": ID_COL, "seg_col": SEG_COL,
            "note_pp_r02": "MODE_B: sem conversao de intermediarios para Perdida",
            "pr_run_id": PR_RUN_ID, "mode_run_id": MODE_RUN_ID,
            "t_pre_proc_model_container_run_id": PREPROC_CONTAINER_RUN_ID,
        })
        mlflow.log_dict(rules_catalog_for_logging(RULES_BY_BLOCK), "rules_catalog.json")

        df_seg_in = spark.table(COTACAO_SEG_FQN)
        mlflow.log_metric("n_seg_in", int(df_seg_in.count()))

        exec_log = {}
        df_seg_pp, exec_log["rules_on_df_seg"] = apply_rules_block("rules_on_df_seg", df_seg_in, RULES_ON_DF_SEG)
        mlflow.log_metric("n_seg_after_rules", int(df_seg_pp.count()))

        df_base, exec_log["rules_build_base"] = apply_rules_block("rules_build_base", df_seg_pp, RULES_BUILD_BASE)
        df_model_tmp, exec_log["rules_build_df_model"]     = apply_rules_block("rules_build_df_model", df_base, RULES_BUILD_MODEL)
        df_valid_tmp, exec_log["rules_build_df_validacao"] = apply_rules_block("rules_build_df_validacao", df_base, RULES_BUILD_VALID)

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
            "fs_decimal_cols":         json.dumps(FS_DECIMAL_COLS),
            "fs_dias_cols":            json.dumps(FS_DIAS_COLS),
            "fs_cat_cols":             json.dumps(FS_CAT_COLS),
            "ensemble_type":           "weighted_auc_pr",
            "mi_parallel":             "true",
            "mi_in_ensemble":          "false",
            "qtd_hr_excluded":         "true",
            "qtd_hr_exclusion_reason": "leakage_pendente",
            "run_suffix":              RUN_SUFFIX,
            "ts_exec":                 TS_EXEC,
            "mode_code":               MODE_CODE,
            "pr_run_id":               PR_RUN_ID,
            "mode_run_id":             MODE_RUN_ID,
            "fs_container_run_id":     FS_CONTAINER_RUN_ID,
        })

        # ── [1] Load + filter SEG_TARGET ──────────────────────────────────
        df_raw = spark.table(DF_MODEL_FQN)
        df_seg = df_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()
        n_rows_seg = int(df_seg.count())
        mlflow.log_metric("n_rows_seg", n_rows_seg)
        if n_rows_seg == 0:
            raise ValueError(f"❌ Nenhuma linha para SEG_TARGET='{SEG_TARGET}' em {DF_MODEL_FQN}")

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

        # ── [3] Perfil de nulos → drop_null_cols ──────────────────────────
        cols_candidate = [c for c in FS_DECIMAL_COLS + FS_DIAS_COLS + FS_CAT_COLS if c in df_base.columns]
        null_counts    = df_base.agg(*[
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in cols_candidate
        ]).collect()[0].asDict()

        null_profile   = [{"col": c, "nulls": int(null_counts[c]), "pct_null": null_counts[c] / n_rows_seg}
                          for c in cols_candidate]
        drop_null_cols = {r["col"] for r in null_profile if r["pct_null"] > NULL_DROP_PCT}
        mlflow.log_dict({"null_profile": null_profile, "drop_null_cols": sorted(drop_null_cols)},
                        "fs_stage1/null_profile.json")

        # ── [4] Cardinalidade + truncagem OUTROS ──────────────────────────
        cat_present  = [c for c in FS_CAT_COLS if c in df_base.columns and c not in drop_null_cols]
        cat_card_pre = [{"col": c, "count_distinct": int(df_base.select(F.countDistinct(c)).collect()[0][0])}
                        for c in cat_present]
        high_card_cols = [r["col"] for r in cat_card_pre if r["count_distinct"] > HIGH_CARD_THRESHOLD]

        for c in high_card_cols:
            df_base = truncate_high_cardinality(df_base, c, top_n=HIGH_CARD_TOP_N, outros=OUTROS_LABEL)

        cat_card_post = [{"col": c, "count_distinct_post": int(df_base.select(F.countDistinct(c)).collect()[0][0])}
                         for c in cat_present]

        mlflow.log_dict({
            "cat_cardinality_pre":  cat_card_pre,
            "high_card_cols":       high_card_cols,
            "high_card_top_n":      HIGH_CARD_TOP_N,
            "outros_label":         OUTROS_LABEL,
            "cat_cardinality_post": cat_card_post,
        }, "fs_stage1/cat_cardinality.json")

        # ── [5] Listas finais de features ─────────────────────────────────
        drop_constant_cols  = {r["col"] for r in cat_card_pre if r["count_distinct"] <= 1}
        DROP_FEATURES_FINAL = set(DROP_FROM_FEATURES) | drop_null_cols | drop_constant_cols

        NUM_COLS_FINAL   = [c for c in FS_DECIMAL_COLS + FS_DIAS_COLS
                            if c in df_base.columns and c not in DROP_FEATURES_FINAL]
        CAT_COLS_FINAL   = [c for c in FS_CAT_COLS
                            if c in df_base.columns and c not in DROP_FEATURES_FINAL]
        FEATURE_COLS_ALL = CAT_COLS_FINAL + NUM_COLS_FINAL

        if not FEATURE_COLS_ALL:
            raise ValueError("❌ FEATURE_COLS_ALL vazio após drops.")

        mlflow.log_dict({
            "null_drop_pct":       NULL_DROP_PCT,
            "high_card_threshold": HIGH_CARD_THRESHOLD,
            "high_card_top_n":     HIGH_CARD_TOP_N,
            "high_card_cols":      high_card_cols,
            "drop_null_cols":      sorted(drop_null_cols),
            "drop_constant_cols":  sorted(drop_constant_cols),
            "drop_features_final": sorted(DROP_FEATURES_FINAL),
            "cat_cols_final":      CAT_COLS_FINAL,
            "num_cols_final":      NUM_COLS_FINAL,
            "features_total":      len(FEATURE_COLS_ALL),
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

        # ── Variáveis para o step TREINO ──────────────────────────────────
        FS_FEATURES_RANKED = features_ranked
        FS_FEATURE_SETS    = feature_sets

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

TREINO_FEATURE_SET_KEY = "top_7"
TREINO_FEATURE_COLS    = FS_FEATURE_SETS[TREINO_FEATURE_SET_KEY]

# Class weight
USE_CLASS_WEIGHT       = "auto"   # "auto" | True | False
CLASS_WEIGHT_THRESHOLD = 0.30

# CV + grid
CV_FOLDS      = 3
CV_SEED       = 42
CV_METRIC     = "areaUnderPR"
GBT_PARAM_GRID = {
    "maxDepth": [4, 6],
    "stepSize": [0.05, 0.1],
    "maxIter":  100,
}

# Threshold operacional
CAPACIDADE_PCT = 0.10   # fraction do hold-out → define K
LIFT_TARGET    = 2.0
BASELINE_MODE  = "conversao_time"   # "taxa_base" | "conversao_time"
CONVERSAO_TIME = 0.3

print("✅ T_TREINO inputs:")
print("• df_model_fqn :", DF_MODEL_FQN)
print("• df_valid_fqn :", DF_VALID_FQN)
print("• feature_set  :", TREINO_FEATURE_SET_KEY, "→", TREINO_FEATURE_COLS)
print("• use_cw       :", USE_CLASS_WEIGHT, "| threshold:", CLASS_WEIGHT_THRESHOLD)
print("• capacidade   :", CAPACIDADE_PCT, "| lift_target:", LIFT_TARGET)
print("• baseline_mode:", BASELINE_MODE, "| conversao_time:", CONVERSAO_TIME)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def build_preprocess_pipeline(cat_cols: list, num_cols: list) -> Pipeline:
    idx_cols  = [f"{c}__idx" for c in cat_cols]
    ohe_cols  = [f"{c}__ohe" for c in cat_cols]
    imp_cols  = [f"{c}__imp" for c in num_cols]
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


def compute_capacity_metrics(
    df_pred: DataFrame,
    n_total: int,
    capacity_pcts: list,
    baseline: float,
    label_col: str = "label",
) -> list:
    """
    Para cada pct em capacity_pcts, calcula Precision@K, Recall@K, FN@K, Lift@K.
    Espera coluna '_p1' (probabilidade da classe positiva).
    """
    n_pos_total = int(df_pred.filter(F.col(label_col) == 1).count())
    w           = Window.orderBy(F.col("_p1").desc())
    df_ranked   = df_pred.withColumn("_rank", F.row_number().over(w))
    results     = []
    for pct in capacity_pcts:
        k          = max(1, int(n_total * pct))
        df_topk    = df_ranked.filter(F.col("_rank") <= k)
        tp         = int(df_topk.filter(F.col(label_col) == 1).count())
        precision  = tp / k if k > 0 else 0.0
        recall     = tp / n_pos_total if n_pos_total > 0 else 0.0
        fn         = n_pos_total - tp
        lift       = precision / baseline if baseline > 0 else 0.0
        threshold  = float(df_ranked.filter(F.col("_rank") == k).select("_p1").collect()[0][0])
        results.append({
            "capacity_pct": pct, "k": k,
            "precision_at_k": round(precision, 4),
            "recall_at_k":    round(recall, 4),
            "fn_at_k":        fn,
            "lift_at_k":      round(lift, 4),
            "threshold":      round(threshold, 6),
        })
    return results


def confusion_matrix_at_threshold(df_pred: DataFrame, threshold: float, label_col: str = "label") -> dict:
    df_dec = df_pred.withColumn("_pred", (F.col("_p1") >= F.lit(threshold)).cast("int"))
    tp = int(df_dec.filter((F.col(label_col) == 1) & (F.col("_pred") == 1)).count())
    fp = int(df_dec.filter((F.col(label_col) == 0) & (F.col("_pred") == 1)).count())
    fn = int(df_dec.filter((F.col(label_col) == 1) & (F.col("_pred") == 0)).count())
    tn = int(df_dec.filter((F.col(label_col) == 0) & (F.col("_pred") == 0)).count())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4)}


print("✅ Helpers T_TREINO carregados")

# COMMAND ----------

RUN_TREINO_EXEC = run_name_vts("T_TREINO")

_tr_kw = {"run_id": TREINO_RUN_ID_OVERRIDE} if TREINO_RUN_ID_OVERRIDE else {"run_name": STEP_TREINO_NAME}

with mlflow.start_run(**_tr_kw, nested=True) as treino_container:
    TREINO_CONTAINER_RUN_ID = treino_container.info.run_id

    with mlflow.start_run(run_name=RUN_TREINO_EXEC, nested=True):

        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "TREINO",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })

        # ── [1] Carga e pré-processamento ─────────────────────────────────
        df_model_raw = spark.table(DF_MODEL_FQN)
        df_valid_raw = spark.table(DF_VALID_FQN)

        df_model_seg = df_model_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))
        df_valid_seg = df_valid_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))

        # Cleaning: blank → null em strings
        for df_name, df_ref in [("model", df_model_seg), ("valid", df_valid_seg)]:
            for c, t in df_ref.dtypes:
                if t == "string":
                    df_ref = df_ref.withColumn(
                        c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c))
                    )
            if df_name == "model":
                df_model_seg = df_ref
            else:
                df_valid_seg = df_ref

        # OUTROS truncation (mesmo threshold/top_n do FS)
        treino_cat_cols = [c for c in TREINO_FEATURE_COLS if c in FS_CAT_COLS]
        treino_num_cols = [c for c in TREINO_FEATURE_COLS if c in FS_DECIMAL_COLS + FS_DIAS_COLS]

        for c in treino_cat_cols:
            card = int(df_model_seg.select(F.countDistinct(c)).collect()[0][0])
            if card > HIGH_CARD_THRESHOLD:
                df_model_seg = truncate_high_cardinality(df_model_seg, c, HIGH_CARD_TOP_N, OUTROS_LABEL)
                df_valid_seg = truncate_high_cardinality(df_valid_seg, c, HIGH_CARD_TOP_N, OUTROS_LABEL)

        for c in treino_num_cols:
            df_model_seg = df_model_seg.withColumn(c, F.col(c).cast("double"))
            df_valid_seg = df_valid_seg.withColumn(c, F.col(c).cast("double"))

        # Class weight
        df_model_seg, label_rate, weight_pos, apply_cw = add_class_weights(
            df_model_seg, LABEL_COL, USE_CLASS_WEIGHT, CLASS_WEIGHT_THRESHOLD
        )
        n_model = int(df_model_seg.count())
        n_valid = int(df_valid_seg.count())

        mlflow.log_params({
            "df_model_fqn":          DF_MODEL_FQN,
            "df_valid_fqn":          DF_VALID_FQN,
            "seg_target":            SEG_TARGET,
            "feature_set":           TREINO_FEATURE_SET_KEY,
            "feature_cols":          json.dumps(TREINO_FEATURE_COLS),
            "n_features":            len(TREINO_FEATURE_COLS),
            "n_model":               n_model,
            "n_valid":               n_valid,
            "use_class_weight":      str(USE_CLASS_WEIGHT),
            "class_weight_threshold": CLASS_WEIGHT_THRESHOLD,
            "label_rate":            round(label_rate, 4),
            "apply_cw":              str(apply_cw),
            "weight_pos":            round(weight_pos, 4),
            "cv_folds":              CV_FOLDS,
            "cv_seed":               CV_SEED,
            "cv_metric":             CV_METRIC,
            "gbt_param_grid":        json.dumps(GBT_PARAM_GRID),
            "gbt_maxiter_fixed":     GBT_PARAM_GRID["maxIter"],
            "mode_code":             MODE_CODE,
            "pr_run_id":             PR_RUN_ID,
            "mode_run_id":           MODE_RUN_ID,
            "treino_container_run_id": TREINO_CONTAINER_RUN_ID,
            "note_calibration":      "defer_v9_threshold_via_capacidade_k",
        })

        df_model_ml = df_model_seg.select(ID_COL, LABEL_COL, "weight", *TREINO_FEATURE_COLS).cache()
        df_valid_ml = df_valid_seg.select(LABEL_COL, *TREINO_FEATURE_COLS).cache()

        # ── [2] CV 3-fold + grid de hiperparâmetros ───────────────────────
        import itertools
        param_combinations = [
            {"maxDepth": d, "stepSize": s, "maxIter": GBT_PARAM_GRID["maxIter"]}
            for d, s in itertools.product(GBT_PARAM_GRID["maxDepth"], GBT_PARAM_GRID["stepSize"])
        ]

        evaluator_pr = BinaryClassificationEvaluator(
            labelCol=LABEL_COL, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
        )

        folds      = kfold_split(df_model_ml, CV_FOLDS, CV_SEED, ID_COL)
        grid_results  = []
        fold_metrics  = []

        for combo in param_combinations:
            combo_key  = f"d{combo['maxDepth']}_s{str(combo['stepSize']).replace('.','')}"
            fold_aucs  = []

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

        best_combo   = max(grid_results, key=lambda x: x["avg_auc_pr"])
        BEST_PARAMS  = {
            "maxDepth": best_combo["maxDepth"],
            "stepSize": best_combo["stepSize"],
            "maxIter":  GBT_PARAM_GRID["maxIter"],
        }

        mlflow.log_param("gbt_params_winner", json.dumps(BEST_PARAMS))
        mlflow.log_dict(grid_results,  "cv/grid_results.json")
        mlflow.log_dict(fold_metrics,  "cv/fold_metrics.json")

        print("✅ CV concluído")
        print("• vencedor :", BEST_PARAMS, f"→ avg AUC-PR={best_combo['avg_auc_pr']:.4f}")

        # ── [3] Treino final em df_model completo ─────────────────────────
        pp_final     = build_preprocess_pipeline(treino_cat_cols, treino_num_cols)
        pp_final_fit = pp_final.fit(df_model_ml)
        df_model_vec = pp_final_fit.transform(df_model_ml).select(LABEL_COL, "weight", "features_vec").cache()

        gbt_final = GBTClassifier(
            featuresCol="features_vec", labelCol=LABEL_COL,
            weightCol="weight" if apply_cw else None,
            maxDepth=BEST_PARAMS["maxDepth"], stepSize=BEST_PARAMS["stepSize"],
            maxIter=BEST_PARAMS["maxIter"], seed=CV_SEED,
        )
        gbt_final_model = gbt_final.fit(df_model_vec)

        mlflow.spark.log_model(gbt_final_model, artifact_path="treino_final/model")

        # ── [4] Avaliação no hold-out ─────────────────────────────────────
        df_valid_vec = pp_final_fit.transform(df_valid_ml).select(LABEL_COL, "features_vec").cache()
        p1_col       = vector_to_array(F.col("probability")).getItem(1).cast("double")
        df_pred      = gbt_final_model.transform(df_valid_vec).withColumn("_p1", p1_col)

        # Métricas globais
        ap_holdout     = compute_ap(df_pred, label_col=LABEL_COL)
        auc_pr_holdout = float(evaluator_pr.evaluate(df_pred))
        mlflow.log_metrics({"ap_holdout": ap_holdout, "auc_pr_holdout": auc_pr_holdout})

        # Baseline
        if BASELINE_MODE == "conversao_time":
            baseline_val = float(CONVERSAO_TIME)
        else:
            baseline_val = float(df_pred.agg(F.avg(F.col(LABEL_COL).cast("double"))).collect()[0][0])

        precision_target_derived = LIFT_TARGET * baseline_val
        mlflow.log_params({
            "capacidade_pct":          CAPACIDADE_PCT,
            "lift_target":             LIFT_TARGET,
            "baseline_mode":           BASELINE_MODE,
            "baseline_calculado":      round(baseline_val, 4),
            "precision_target_derivado": round(precision_target_derived, 4),
        })
        if BASELINE_MODE == "conversao_time":
            mlflow.log_param("conversao_time", CONVERSAO_TIME)

        # Curva de capacidade: K = [5%, 10%, 20%] + CAPACIDADE_PCT
        capacity_pcts = sorted(set([0.05, 0.10, 0.20, CAPACIDADE_PCT]))
        cap_metrics   = compute_capacity_metrics(df_pred, n_valid, capacity_pcts, baseline_val, LABEL_COL)

        op_row        = next(r for r in cap_metrics if r["capacity_pct"] == CAPACIDADE_PCT)
        threshold_op  = op_row["threshold"]

        mlflow.log_params({
            "k_operacional":     op_row["k"],
            "threshold_operacional": threshold_op,
        })
        mlflow.log_metrics({
            "precision_at_k_op": op_row["precision_at_k"],
            "recall_at_k_op":    op_row["recall_at_k"],
            "fn_at_k_op":        op_row["fn_at_k"],
            "lift_at_k_op":      op_row["lift_at_k"],
        })
        for r in cap_metrics:
            pct_label = str(int(r["capacity_pct"] * 100))
            mlflow.log_metrics({
                f"precision_at_{pct_label}pct": r["precision_at_k"],
                f"recall_at_{pct_label}pct":    r["recall_at_k"],
                f"fn_at_{pct_label}pct":        r["fn_at_k"],
                f"lift_at_{pct_label}pct":      r["lift_at_k"],
            })

        # Matriz de confusão com threshold operacional
        cm = confusion_matrix_at_threshold(df_pred, threshold_op, LABEL_COL)

        # Artifacts
        cap_pdf = pd.DataFrame(cap_metrics)
        log_pandas_csv(cap_pdf, "threshold_analysis/capacity_curve.csv")
        mlflow.log_dict(cm, "threshold_analysis/confusion_matrix.json")
        mlflow.log_dict({
            "feature_set":    TREINO_FEATURE_SET_KEY,
            "feature_cols":   TREINO_FEATURE_COLS,
            "best_params":    BEST_PARAMS,
            "label_rate":     round(label_rate, 4),
            "apply_cw":       apply_cw,
            "weight_pos":     round(weight_pos, 4),
            "baseline":       round(baseline_val, 4),
            "precision_target_derived": round(precision_target_derived, 4),
            "ap_holdout":     round(ap_holdout, 4),
            "auc_pr_holdout": round(auc_pr_holdout, 4),
            "threshold_op":   threshold_op,
            "capacity_metrics": cap_metrics,
            "confusion_matrix": cm,
        }, "threshold_analysis/metrics_summary.json")

        df_model_vec.unpersist(); df_valid_vec.unpersist()

        print("✅ T_TREINO ok")
        print("• best_params   :", BEST_PARAMS)
        print("• ap_holdout    :", round(ap_holdout, 4))
        print("• auc_pr_holdout:", round(auc_pr_holdout, 4))
        print("• threshold_op  :", threshold_op, f"(K={op_row['k']}, cap={CAPACIDADE_PCT*100:.0f}%)")
        print("• precision@K   :", op_row["precision_at_k"])
        print("• recall@K      :", op_row["recall_at_k"])
        print("• lift@K        :", op_row["lift_at_k"])
        print("• fn@K          :", op_row["fn_at_k"])