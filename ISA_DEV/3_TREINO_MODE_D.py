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
EXPERIMENT_NAME = "/Users/psw.service@pswdigital.com.br/ISA_DEV/ISA_EXP"  # <<< AJUSTE

PR_TREINO_NAME = "T_PR_TREINO"
MODE_CODE      = "D"
MODE_NAME      = f"T_MODE_{MODE_CODE}"

PR_RUN_ID_OVERRIDE   = "21f1184afb354159b325ad87bca3c50d"
MODE_RUN_ID_OVERRIDE = "bb141e3ebebe418792b1eaf0e11470ea"

STEP_PRE_PROC_NAME    = "T_PRE_PROC_MODEL"
STEP_CLF_EXPLORE_NAME = "T_CLUSTERING_EXPLORE"
STEP_CLF_FIT_NAME     = "T_CLUSTERING_FIT"

# =========================
# Versionamento
# =========================
TREINO_VERSAO            = "V11.0.0"
TREINO_VERSAO_TABLE_SAFE = TREINO_VERSAO.replace(".", "_")
VERSAO_REF               = TREINO_VERSAO

TS_EXEC  = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_UUID = uuid.uuid4().hex[:8]

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# =========================
# INPUT
# =========================
COTACAO_SEG_FQN = "silver.cotacao_seg_20260326_105205"  # <<< AJUSTE

# =========================
# OUTPUT — criado apenas em T_CLUSTERING_FIT
# =========================
OUT_SCHEMA   = "gold"
DF_MODEL_FQN = f"{OUT_SCHEMA}.cotacao_model_d_{TS_EXEC}_{RUN_UUID}"
DF_VALID_FQN = f"{OUT_SCHEMA}.cotacao_validacao_d_{TS_EXEC}_{RUN_UUID}"
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
SPLIT_SALT           = "split_c1_seg_mes"  # <<< mesmo do MODE_C — mantido para comparabilidade

DO_PROFILE = True

# =========================
# PRE_PROC_MODEL / FS — colunas e segmento
# =========================
SEG_TARGET = "SEGURO_NOVO_MANUAL"  # <<< AJUSTE

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
# CLF_CORRETOR adicionado como feature categórica derivada por clustering.
FEATURE_CANDIDATES = {
    "VL_PREMIO_ALVO":                            True,   # decimal(17,2)
    "INTERMENDIARIO_PERFIL":                     True,   # string
    "DT_INICIO_VIGENCIA":                        False,  # date
    "VL_PREMIO_LIQUIDO":                         True,   # decimal(17,2)
    "VL_PRE_TOTAL":                              True,   # decimal(17,2)
    "DS_PRODUTO_NOME":                           True,   # string
    "DS_SISTEMA":                                False,  # string
    "VL_ENDOSSO_PREMIO_TOTAL":                   True,   # decimal(17,2)
    "CD_FILIAL_RESPONSAVEL_COTACAO":             True,   # string
    "DS_ATIVIDADE_SEGURADO":                     True,   # string
    "DS_GRUPO_CORRETOR_SEGMENTO":                False,  # string
    "DIAS_ULTIMA_ATUALIZACAO":                   True,   # int
    "DIAS_VALIDADE":                             True,   # int
    "DIAS_ANALISE_SUBSCRICAO":                   True,   # int
    "DIAS_FIM_ANALISE_SUBSCRICAO":               True,   # int
    "DIAS_COTACAO":                              True,   # int
    "DIAS_INICIO_VIGENCIA":                      True,   # int
    "VL_GWP_CORRETOR_resumo":                    False,  # decimal(17,2)
    "QTD_ACORDO_COMERCIAL_resumo":               False,  # bigint
    "QTD_COTACAO_2025_detalhe":                  True,   # bigint
    "QTD_EMITIDO_2025_detalhe":                  True,   # bigint
    "HR_2025_detalhe":                           True,   # decimal(17,6)
    "CLF_CORRETOR":                              True,   # string — cluster K-Means do corretor (MODE_D)
}

# =========================
# CLUSTERING — params
# =========================
# CLUSTER_SEG_FILTER: None = perfil global (todos os SEGs) | SEG_TARGET = perfil restrito ao SEG atual
CLUSTER_SEG_FILTER  = None  # <<< AJUSTE — None ou SEG_TARGET
CLF_RANDOM_SEED     = 42
CLF_NULL_STRATEGY   = "drop"            # "drop" | "impute_median"
CLF_FEATURES        = ["hr_mean", "cotacao_mean", "n_produtos"]
CLF_K_RANGE_EXPLORE = [2, 3, 4, 5, 6, 7]  # <<< AJUSTE — range para elbow + silhouette
CLF_SCALE_STRATEGIES = ["standard", "log1p_standard", "robust"]  # exploradas simultaneamente no EXPLORE

print("✅ CONFIG MODE_D carregada")
print("• input               :", COTACAO_SEG_FQN)
print("• mode                :", MODE_CODE)
print("• versao              :", TREINO_VERSAO)
print("• seg_target          :", SEG_TARGET)
print("• split_salt          :", SPLIT_SALT)
print("• clf_k_range_explore :", CLF_K_RANGE_EXPLORE)
print("• clf_scale_strategies:", CLF_SCALE_STRATEGIES)
print("• clf_null_strategy   :", CLF_NULL_STRATEGY)
print("• cluster_seg_filter  :", CLUSTER_SEG_FILTER)
print("• df_model_fqn        :", DF_MODEL_FQN)
print("• df_valid_fqn        :", DF_VALID_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspeção de colunas candidatas
# MAGIC Executar esta célula para gerar `FEATURE_CANDIDATES` a partir da tabela.
# MAGIC Copie o output na Config acima e ajuste os toggles antes de rodar o pipeline.
# MAGIC Lembrar de adicionar `"CLF_CORRETOR": True` manualmente — ela não existe em cotacao_seg.

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
print('    "CLF_CORRETOR":                          True,   # string — cluster K-Means do corretor (MODE_D)')
print("}")
print()
print(f"# Candidatas: {len(_candidates) + 1} | Excluídas estruturalmente ({len(_never_feat)}): {sorted(_never_feat)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers globais

# COMMAND ----------

import io
import json
import os
import pickle
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


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


def compute_top_vals(df: DataFrame, col_name: str, top_n: int) -> list:
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
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(top_vals), F.col(col_name)).otherwise(F.lit(outros)),
    )


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
        "outputs":       {},
        "note":          "df_model e df_validacao permanecem em memória — salvos em gold por T_CLUSTERING_FIT",
    }

def build_tables_lineage_clf_fit() -> dict:
    return {
        "stage":         "T_CLUSTERING_FIT",
        "ts_exec":       TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode":          MODE_CODE,
        "inputs":        {"cotacao_seg_perfil": COTACAO_SEG_FQN, "df_model_in_memory": True},
        "outputs":       {"df_model": DF_MODEL_FQN, "df_validacao": DF_VALID_FQN},
    }


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
    # MODE_D: sem conversão de status intermediários para Perdida.
    return df.filter(F.col(STATUS_COL).isin(ALLOWED_FINAL_STATUS))

def PP_R03_cria_label(df: DataFrame) -> DataFrame:
    return df.withColumn(LABEL_COL,
        F.when(F.col(STATUS_COL) == "Emitida", F.lit(1.0))
         .when(F.col(STATUS_COL) == "Perdida", F.lit(0.0))
         .otherwise(F.lit(None).cast("double")))

MESES_EXCLUSAO_PP_R07 = ["2025-11", "2025-12"]

def PP_R07_drop_meses_exclusao(df: DataFrame) -> DataFrame:
    mes_col = F.date_format(F.to_date(F.col(DATE_COL)), "yyyy-MM")
    return df.filter(~mes_col.isin(MESES_EXCLUSAO_PP_R07))

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

TOGGLES_RULES_ON_DF_SEG = {
    "PP_R01": True,   # Normalizar DS_GRUPO_STATUS (EMITIDA→Emitida, PERDIDA→Perdida)
    "PP_R02": True,   # Manter apenas status finais (Emitida, Perdida) — sem converter intermediários
    "PP_R03": True,   # Criar coluna label (Emitida=1.0, Perdida=0.0)
    "PP_R07": True,   # Excluir cotações dos meses 2025-11 e 2025-12
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
    rule_def("PP_R07", f"Excluir cotações dos meses {MESES_EXCLUSAO_PP_R07}",
             PP_R07_drop_meses_exclusao,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R07"],
             requires_columns=[DATE_COL]),
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

with mlflow.start_run(run_name=STEP_PRE_PROC_NAME, nested=True) as _pp_container:
    PREPROC_CONTAINER_RUN_ID = _pp_container.info.run_id

    with mlflow.start_run(run_name=run_name_vts(STEP_PRE_PROC_NAME), nested=True):
        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "PRE_PROC_MODEL",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })
        mlflow.log_params({
            "ts_exec": TS_EXEC, "treino_versao": TREINO_VERSAO, "mode_code": MODE_CODE,
            "seg_target": SEG_TARGET,
            "input_cotacao_seg_fqn": COTACAO_SEG_FQN,
            "valid_frac": float(VALID_FRAC), "split_salt": SPLIT_SALT,
            "allowed_final_status": json.dumps(ALLOWED_FINAL_STATUS),
            "label_col": LABEL_COL, "id_col": ID_COL, "seg_col": SEG_COL,
            "note_pp_r02": "MODE_D: sem conversao de intermediarios para Perdida",
            "pr_run_id": PR_RUN_ID, "mode_run_id": MODE_RUN_ID,
            "t_pre_proc_model_container_run_id": PREPROC_CONTAINER_RUN_ID,
        })
        mlflow.log_dict(rules_catalog_for_logging(RULES_BY_BLOCK), "rules_catalog.json")
        mlflow.log_dict(build_tables_lineage_preproc(), "tables_lineage.json")

        df_seg_in = spark.table(COTACAO_SEG_FQN)
        n_seg_in  = int(df_seg_in.count())
        mlflow.log_metric("n_seg_in", n_seg_in)

        # ── rules_on_df_seg — aplicar individualmente para logar n_linhas_por_regra ──
        df_seg_pp    = df_seg_in
        exec_log     = {}
        exec_log_seg = []
        for r in RULES_ON_DF_SEG:
            rid, desc = r["rule_id"], r["description"]
            req        = r.get("requires_columns", []) or []
            is_enabled = bool(r.get("enabled", True)) and bool(TOGGLES_RULES_ON_DF_SEG.get(rid, True))
            if not is_enabled:
                exec_log_seg.append({"rule_id": rid, "status": "SKIPPED_DISABLED", "description": desc})
                continue
            missing = [c for c in req if c not in df_seg_pp.columns]
            if missing:
                exec_log_seg.append({"rule_id": rid, "status": "SKIPPED_MISSING_COLS", "description": desc,
                                     "reason": f"missing={missing}"})
                continue
            df_seg_pp = r["fn"](df_seg_pp)
            n_after   = int(df_seg_pp.count())
            mlflow.log_metric(f"n_linhas_por_regra_{rid}", n_after)
            exec_log_seg.append({"rule_id": rid, "status": "APPLIED", "description": desc,
                                  "n_rows_after": n_after})

        exec_log["rules_on_df_seg"] = exec_log_seg
        mlflow.log_metric("n_seg_after_rules", int(df_seg_pp.count()))

        df_base, exec_log["rules_build_base"] = apply_rules_block("rules_build_base", df_seg_pp, RULES_BUILD_BASE)

        # ── Split ─────────────────────────────────────────────────────────────────
        df_model_tmp, exec_log["rules_build_df_model"]     = apply_rules_block("rules_build_df_model", df_base, RULES_BUILD_MODEL)
        df_valid_tmp, exec_log["rules_build_df_validacao"] = apply_rules_block("rules_build_df_validacao", df_base, RULES_BUILD_VALID)

        # ── PP_R04 / PP_R05 / PP_R06 — feature prep ───────────────────────────────
        _schema_pp      = dict(df_model_tmp.dtypes)
        _fc_all_enabled = [c for c, v in FEATURE_CANDIDATES.items() if v and c in _schema_pp]
        _pp_cat_cols    = [c for c in _fc_all_enabled if _schema_pp[c] == "string"]
        _pp_num_cols    = [c for c in _fc_all_enabled if _schema_pp[c] != "string"]
        _pp_all_cand    = _pp_cat_cols + _pp_num_cols

        n_rows_pp = int(df_model_tmp.count())

        # PP_R04 — remoção de features com >NULL_DROP_PCT nulos
        null_counts_pp = df_model_tmp.agg(*[
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
        _fp_desc = {r["rule_id"]: r["description"] for r in RULES_FEATURE_PREP}
        mlflow.log_dict(
            [
                {"rule_id": "PP_R04", "description": _fp_desc["PP_R04"],
                 "status": exec_log["PP_R04"]["status"], "null_profile": null_profile_pp,
                 "drop_null_cols": drop_null_cols_pp},
                {"rule_id": "PP_R05", "description": _fp_desc["PP_R05"],
                 "status": exec_log["PP_R05"]["status"], "high_card_cols": high_card_cols_pp,
                 "top_vals_by_col": top_vals_by_col_prep},
                {"rule_id": "PP_R06", "description": _fp_desc["PP_R06"],
                 "status": exec_log["PP_R06"]["status"], "drop_constant_cols": drop_const_cols_pp},
            ],
            "preproc_feature/rules_feature_prep.json",
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

        # df_model_tmp e df_valid_tmp permanecem em memória — salvos em gold por T_CLUSTERING_FIT

print("✅ T_PRE_PROC_MODEL ok — df_model e df_validacao em memória")
print("• df_model    :", n_model, "linhas")
print("• df_validacao:", n_valid, "linhas")
print("• PREPROC_CONTAINER_RUN_ID:", PREPROC_CONTAINER_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_CLUSTERING_EXPLORE

# COMMAND ----------

with mlflow.start_run(run_name=STEP_CLF_EXPLORE_NAME, nested=True) as _clf_explore_container:
    CLF_EXPLORE_CONTAINER_RUN_ID = _clf_explore_container.info.run_id

    with mlflow.start_run(run_name=run_name_vts(STEP_CLF_EXPLORE_NAME), nested=True):
        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "CLUSTERING_EXPLORE",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })
        mlflow.log_params({
            "clf_k_range_explore":    json.dumps(CLF_K_RANGE_EXPLORE),
            "clf_scale_strategies":   json.dumps(CLF_SCALE_STRATEGIES),
            "clf_random_seed":        CLF_RANDOM_SEED,
            "clf_null_strategy":      CLF_NULL_STRATEGY,
            "clf_cluster_features":   json.dumps(CLF_FEATURES),
            "clf_cluster_seg_filter": str(CLUSTER_SEG_FILTER),
            "cotacao_seg_fqn":        COTACAO_SEG_FQN,
            "pr_run_id":              PR_RUN_ID,
            "mode_run_id":            MODE_RUN_ID,
        })

        # ── Load e agregação do cotacao_seg ───────────────────────────────────────
        df_perfil = spark.table(COTACAO_SEG_FQN)
        if CLUSTER_SEG_FILTER is not None:
            df_perfil = df_perfil.filter(F.col(SEG_COL) == F.lit(CLUSTER_SEG_FILTER))

        pdf_perfil = (
            df_perfil
            .select("CD_DOC_CORRETOR", "DS_PRODUTO_NOME", "HR_2025_detalhe", "QTD_COTACAO_2025_detalhe")
            .groupBy("CD_DOC_CORRETOR")
            .agg(
                F.mean("HR_2025_detalhe").alias("hr_mean"),
                F.mean("QTD_COTACAO_2025_detalhe").alias("cotacao_mean"),
                F.countDistinct("DS_PRODUTO_NOME").alias("n_produtos"),
            )
            .toPandas()
        )

        n_corretores_total = len(pdf_perfil)
        clf_null_counts    = {c: int(pdf_perfil[c].isnull().sum()) for c in CLF_FEATURES}
        clf_null_pct       = {c: round(clf_null_counts[c] / n_corretores_total, 4) for c in CLF_FEATURES}
        mlflow.log_metric("clf_n_corretores_total", n_corretores_total)
        mlflow.log_dict({"null_counts": clf_null_counts, "null_pct": clf_null_pct}, "clustering/null_profile.json")

        # ── NULL handling ─────────────────────────────────────────────────────────
        if CLF_NULL_STRATEGY == "drop":
            pdf_fit = pdf_perfil.dropna(subset=CLF_FEATURES).copy()
        elif CLF_NULL_STRATEGY == "impute_median":
            pdf_fit = pdf_perfil.copy()
            for _c in CLF_FEATURES:
                pdf_fit[_c] = pdf_fit[_c].fillna(pdf_fit[_c].median())
        else:
            raise ValueError(f"❌ CLF_NULL_STRATEGY inválido: {CLF_NULL_STRATEGY}")

        n_corretores_clustered = len(pdf_fit)
        mlflow.log_metric("clf_n_corretores_clustered", n_corretores_clustered)

        X_raw = pdf_fit[CLF_FEATURES].values

        # ── Distribuições brutas — antes de qualquer normalização ─────────────────
        # DS_PRODUTO_NOME: não é feature direta — é a fonte de n_produtos (countDistinct).
        # Plotamos um bar chart de corretores por produto para contextualizar n_produtos.
        pdf_produto_dist = (
            df_perfil
            .groupBy("DS_PRODUTO_NOME")
            .agg(F.countDistinct("CD_DOC_CORRETOR").alias("n_corretores"))
            .orderBy(F.col("n_corretores").desc())
            .limit(20)
            .toPandas()
        )

        with tempfile.TemporaryDirectory() as _tmpdir_raw:
            # Bar chart: DS_PRODUTO_NOME (contexto para n_produtos)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                pdf_produto_dist["DS_PRODUTO_NOME"].iloc[::-1].values,
                pdf_produto_dist["n_corretores"].iloc[::-1].values,
                color="steelblue",
            )
            ax.set_xlabel("N corretores distintos")
            ax.set_title("Corretores por produto (top 20) — DS_PRODUTO_NOME [contexto de n_produtos]")
            ax.grid(axis="x", alpha=0.3)
            _p = os.path.join(_tmpdir_raw, "dist_ds_produto_nome.png")
            fig.savefig(_p, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(_p, artifact_path="clustering/explore/raw")

            # Histogramas: hr_mean, cotacao_mean, n_produtos
            _pct_levels  = [10, 25, 50, 75, 90]
            _pct_colors  = {10: "red", 25: "orange", 50: "green", 75: "orange", 90: "red"}
            _pct_ls      = {10: "--",  25: "--",     50: "-",     75: "--",     90: "--"}
            for _col in CLF_FEATURES:
                _vals = pdf_fit[_col].dropna().values
                _pcts = {p: float(np.percentile(_vals, p)) for p in _pct_levels}
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(_vals, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
                for _p_lvl, _p_val in _pcts.items():
                    ax.axvline(_p_val, color=_pct_colors[_p_lvl], linestyle=_pct_ls[_p_lvl],
                               linewidth=1.2, label=f"p{_p_lvl}={_p_val:.2f}")
                ax.set_xlabel(_col)
                ax.set_ylabel("Frequência")
                ax.set_title(f"Distribuição bruta — {_col}")
                ax.legend(fontsize=8)
                _p = os.path.join(_tmpdir_raw, f"dist_raw_{_col}.png")
                fig.savefig(_p, bbox_inches="tight", dpi=120)
                plt.close(fig)
                mlflow.log_artifact(_p, artifact_path="clustering/explore/raw")

        # def _build_scaler_and_X(X, strategy):
        #     if strategy == "standard":
        #         sc = StandardScaler()
        #         return sc, sc.fit_transform(X)
        #     elif strategy == "log1p_standard":
        #         sc = StandardScaler()
        #         return sc, sc.fit_transform(np.log1p(X))
        #     elif strategy == "robust":
        #         sc = RobustScaler()
        #         return sc, sc.fit_transform(X)
        #     else:
        #         raise ValueError(f"❌ CLF_SCALE_STRATEGY inválido: {strategy}")

        def _build_scaler_and_X(X, strategy):
            if strategy == "standard":
                sc = StandardScaler()
                return sc, sc.fit_transform(X)
            elif strategy == "log1p_standard":
                sc = StandardScaler()
                # Convert to float before applying log1p
                X_float = X.astype(float)
                return sc, sc.fit_transform(np.log1p(X_float))
            elif strategy == "robust":
                sc = RobustScaler()
                return sc, sc.fit_transform(X)
            else:
                raise ValueError(f"❌ CLF_SCALE_STRATEGY inválido: {strategy}")

        # ── K-Means explore — loop sobre estratégias × K ──────────────────────────
        explore_by_strategy: dict = {}

        for _strategy in CLF_SCALE_STRATEGIES:
            print(f"\n── estratégia: {_strategy} ──")
            _sc, _X = _build_scaler_and_X(X_raw, _strategy)

            # ── Distribuições pós-normalização ────────────────────────────────────
            with tempfile.TemporaryDirectory() as _tmpdir_dist:
                for _j, _col in enumerate(CLF_FEATURES):
                    _vals_sc = _X[:, _j]
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(_vals_sc, bins=50, color="darkorange", edgecolor="none", alpha=0.8)
                    ax.axvline(float(np.mean(_vals_sc)),   color="green", linestyle="-",  linewidth=1.5,
                               label=f"mean={np.mean(_vals_sc):.3f}")
                    ax.axvline(float(np.median(_vals_sc)), color="blue",  linestyle="--", linewidth=1.5,
                               label=f"median={np.median(_vals_sc):.3f}")
                    ax.set_xlabel(f"{_col} (pós {_strategy})")
                    ax.set_ylabel("Frequência")
                    ax.set_title(f"Distribuição pós-normalização — {_col} [{_strategy}]")
                    ax.legend(fontsize=8)
                    _p = os.path.join(_tmpdir_dist, f"dist_scaled_{_col}.png")
                    fig.savefig(_p, bbox_inches="tight", dpi=120)
                    plt.close(fig)
                    mlflow.log_artifact(_p, artifact_path=f"clustering/explore/{_strategy}")

            _inertias, _silhouettes = [], []

            for _k in CLF_K_RANGE_EXPLORE:
                _km = KMeans(n_clusters=_k, n_init=10, random_state=CLF_RANDOM_SEED)
                _km.fit(_X)
                _inertia    = float(_km.inertia_)
                _silhouette = float(silhouette_score(_X, _km.labels_)) if _k > 1 else 0.0
                _inertias.append(_inertia)
                _silhouettes.append(_silhouette)
                print(f"  K={_k}: inertia={_inertia:.1f} | silhouette={_silhouette:.4f}")

            _results = [
                {"k": k, "inertia": inertia, "silhouette": sil}
                for k, inertia, sil in zip(CLF_K_RANGE_EXPLORE, _inertias, _silhouettes)
            ]
            explore_by_strategy[_strategy] = {"scaler": _sc, "X_scaled": _X, "results": _results}

            mlflow.log_dict(_results, f"clustering/explore/{_strategy}/explore_results.json")

            with tempfile.TemporaryDirectory() as _tmpdir:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(CLF_K_RANGE_EXPLORE, _inertias, marker="o", linewidth=2)
                ax.set_xlabel("K")
                ax.set_ylabel("Inertia")
                ax.set_title(f"Elbow Curve — {_strategy}")
                ax.grid(True, alpha=0.3)
                _p = os.path.join(_tmpdir, "elbow_curve.png")
                fig.savefig(_p, bbox_inches="tight", dpi=120)
                plt.close(fig)
                mlflow.log_artifact(_p, artifact_path=f"clustering/explore/{_strategy}")

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(CLF_K_RANGE_EXPLORE, _silhouettes, marker="o", linewidth=2, color="orange")
                ax.set_xlabel("K")
                ax.set_ylabel("Silhouette Score")
                ax.set_title(f"Silhouette Score vs K — {_strategy}")
                ax.grid(True, alpha=0.3)
                _p = os.path.join(_tmpdir, "silhouette_curve.png")
                fig.savefig(_p, bbox_inches="tight", dpi=120)
                plt.close(fig)
                mlflow.log_artifact(_p, artifact_path=f"clustering/explore/{_strategy}")

print("✅ T_CLUSTERING_EXPLORE ok")
print(f"• corretores para clustering: {n_corretores_clustered} (de {n_corretores_total} total)")
print("• K_RANGE_EXPLORE:", CLF_K_RANGE_EXPLORE)
print("• Estratégias exploradas:", CLF_SCALE_STRATEGIES)
for _strategy in CLF_SCALE_STRATEGIES:
    print(f"\n  [{_strategy}]")
    for r in explore_by_strategy[_strategy]["results"]:
        print(f"    K={r['k']}: inertia={r['inertia']:.1f} | silhouette={r['silhouette']:.4f}")
print("\n• Inspecionar clustering/explore/{strategy}/ no MLflow antes de preencher a célula de decisão")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Célula de decisão — K_FINAL + SCALE_STRATEGY_FINAL
# MAGIC Inspecionar `clustering/explore/{strategy}/elbow_curve.png` e `silhouette_curve.png` no MLflow antes de preencher.

# COMMAND ----------

# ============================================================
# DECISION CELL — preencher após inspecionar MLflow
# ============================================================
CLF_K_FINAL              = 7                  # <<< AJUSTE após ver elbow + silhouette por estratégia
CLF_SCALE_STRATEGY_FINAL = "log1p_standard"   # <<< AJUSTE — "standard" | "log1p_standard" | "robust"

assert CLF_SCALE_STRATEGY_FINAL in CLF_SCALE_STRATEGIES, \
    f"❌ CLF_SCALE_STRATEGY_FINAL='{CLF_SCALE_STRATEGY_FINAL}' não está em CLF_SCALE_STRATEGIES={CLF_SCALE_STRATEGIES}"

print(f"CLF_K_FINAL              = {CLF_K_FINAL}")
print(f"CLF_SCALE_STRATEGY_FINAL = {CLF_SCALE_STRATEGY_FINAL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_CLUSTERING_FIT

# COMMAND ----------

if table_exists(DF_MODEL_FQN):
    raise ValueError(f"❌ DF_MODEL já existe: {DF_MODEL_FQN}")
if table_exists(DF_VALID_FQN):
    raise ValueError(f"❌ DF_VALID já existe: {DF_VALID_FQN}")

with mlflow.start_run(run_name=STEP_CLF_FIT_NAME, nested=True) as _clf_fit_container:
    CLF_FIT_CONTAINER_RUN_ID = _clf_fit_container.info.run_id

    with mlflow.start_run(run_name=run_name_vts(STEP_CLF_FIT_NAME), nested=True):
        mlflow.set_tags({
            "pipeline_tipo": "T", "stage": "TREINO", "run_role": "exec",
            "mode": MODE_CODE, "step": "CLUSTERING_FIT",
            "treino_versao": TREINO_VERSAO, "versao_ref": VERSAO_REF,
        })
        # ── Recuperar scaler + X_scaled da estratégia escolhida ──────────────────
        scaler   = explore_by_strategy[CLF_SCALE_STRATEGY_FINAL]["scaler"]
        X_scaled = explore_by_strategy[CLF_SCALE_STRATEGY_FINAL]["X_scaled"]

        mlflow.log_params({
            "clf_k_final":              CLF_K_FINAL,
            "clf_scale_strategy_final": CLF_SCALE_STRATEGY_FINAL,
            "df_model_fqn":             DF_MODEL_FQN,
            "df_valid_fqn":             DF_VALID_FQN,
            "pr_run_id":                PR_RUN_ID,
            "mode_run_id":              MODE_RUN_ID,
        })
        mlflow.log_dict(build_tables_lineage_clf_fit(), "tables_lineage.json")

        # ── Fit K-Means final (reutiliza scaler + X_scaled do EXPLORE) ────────────
        km = KMeans(n_clusters=CLF_K_FINAL, n_init=10, random_state=CLF_RANDOM_SEED)
        km.fit(X_scaled)

        clf_silhouette = float(silhouette_score(X_scaled, km.labels_)) if CLF_K_FINAL > 1 else 0.0
        clf_inertia    = float(km.inertia_)

        # ── Atribuição de clusters ────────────────────────────────────────────────
        pdf_fit_clf = pdf_fit.copy()
        pdf_fit_clf["CLF_CORRETOR"] = km.labels_.astype(str)

        clf_cluster_dist = {
            f"clf_cluster_{i}_n_corretores": int((pdf_fit_clf["CLF_CORRETOR"] == str(i)).sum())
            for i in range(CLF_K_FINAL)
        }

        mlflow.log_metrics({
            "clf_silhouette_score":         clf_silhouette,
            "clf_inertia":                  clf_inertia,
            "clf_n_corretores_total":       n_corretores_total,
            "clf_n_corretores_clustered":   n_corretores_clustered,
            "clf_n_corretores_sem_cluster": n_corretores_total - n_corretores_clustered,
            **clf_cluster_dist,
        })

        # ── Join CLF_CORRETOR → df_model + df_validacao ───────────────────────────
        cluster_map_sdf = spark.createDataFrame(
            pdf_fit_clf[["CD_DOC_CORRETOR", "CLF_CORRETOR"]]
        )
        df_model_out = df_model_tmp.join(cluster_map_sdf, on="CD_DOC_CORRETOR", how="left")
        df_valid_out = df_valid_tmp.join(cluster_map_sdf, on="CD_DOC_CORRETOR", how="left")

        n_model_total = int(df_model_out.count())
        n_model_clf   = int(df_model_out.filter(F.col("CLF_CORRETOR").isNotNull()).count())
        n_valid_total = int(df_valid_out.count())
        n_valid_clf   = int(df_valid_out.filter(F.col("CLF_CORRETOR").isNotNull()).count())

        pct_cob_model = round(n_model_clf / n_model_total, 4) if n_model_total > 0 else 0.0
        pct_cob_valid = round(n_valid_clf / n_valid_total, 4) if n_valid_total > 0 else 0.0

        mlflow.log_metrics({
            "clf_n_cotacoes_com_clf_model": n_model_clf,
            "clf_pct_cobertura_model":      pct_cob_model,
            "clf_n_cotacoes_com_clf_valid": n_valid_clf,
            "clf_pct_cobertura_valid":      pct_cob_valid,
        })

        if pct_cob_model < 0.80:
            print(f"⚠️  pct_cobertura_model={pct_cob_model:.1%} — abaixo de 80%. Revisar CLF_NULL_STRATEGY.")

        # ── Artifacts ─────────────────────────────────────────────────────────────
        # Para log1p_standard: inverse_transform devolve escala log1p — aplicar expm1 para escala real
        _centroids_inv = scaler.inverse_transform(km.cluster_centers_)
        centroids_original = np.expm1(_centroids_inv) if CLF_SCALE_STRATEGY_FINAL == "log1p_standard" else _centroids_inv

        cluster_summary = [
            {
                "cluster":      str(i),
                "n_corretores": int((pdf_fit_clf["CLF_CORRETOR"] == str(i)).sum()),
                **{f"{feat}_centroid": float(centroids_original[i][j]) for j, feat in enumerate(CLF_FEATURES)},
                **{f"{feat}_median":   float(pdf_fit_clf.loc[pdf_fit_clf["CLF_CORRETOR"] == str(i), feat].median())
                   for feat in CLF_FEATURES},
            }
            for i in range(CLF_K_FINAL)
        ]

        mlflow.log_dict(cluster_summary, "clustering/cluster_summary.json")

        with tempfile.TemporaryDirectory() as _tmpdir:
            # scaler.pkl e kmeans_model.pkl
            _scaler_path = os.path.join(_tmpdir, "scaler.pkl")
            with open(_scaler_path, "wb") as _f:
                pickle.dump(scaler, _f)
            mlflow.log_artifact(_scaler_path, artifact_path="clustering")

            _km_path = os.path.join(_tmpdir, "kmeans_model.pkl")
            with open(_km_path, "wb") as _f:
                pickle.dump(km, _f)
            mlflow.log_artifact(_km_path, artifact_path="clustering")

            # cluster_counts.png
            _clusters = [str(i) for i in range(CLF_K_FINAL)]
            _counts   = [clf_cluster_dist[f"clf_cluster_{i}_n_corretores"] for i in range(CLF_K_FINAL)]
            fig, ax = plt.subplots(figsize=(max(5, CLF_K_FINAL * 1.2), 4))
            ax.bar(_clusters, _counts, color="steelblue")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("N corretores")
            ax.set_title(f"Corretores por cluster (K={CLF_K_FINAL})")
            for _i, _v in enumerate(_counts):
                ax.text(_i, _v + max(_counts) * 0.01, str(_v), ha="center", fontsize=9)
            _path = os.path.join(_tmpdir, "cluster_counts.png")
            fig.savefig(_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(_path, artifact_path="clustering")

            # cluster_heatmap.png
            _heatmap_df = pd.DataFrame(
                centroids_original,
                columns=CLF_FEATURES,
                index=[f"cluster_{i}" for i in range(CLF_K_FINAL)],
            )
            fig, ax = plt.subplots(figsize=(7, max(3, CLF_K_FINAL)))
            sns.heatmap(_heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5)
            ax.set_title(f"Perfil de cluster — centroides em escala original (K={CLF_K_FINAL})")
            _path = os.path.join(_tmpdir, "cluster_heatmap.png")
            fig.savefig(_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(_path, artifact_path="clustering/original")

            # scatter plots (3 combinações)
            _colors = plt.cm.tab10.colors
            _scatter_pairs = [
                ("hr_mean",      "cotacao_mean", "scatter_hr_cotacao.png"),
                ("hr_mean",      "n_produtos",   "scatter_hr_produtos.png"),
                ("cotacao_mean", "n_produtos",   "scatter_cotacao_produtos.png"),
            ]
            for _x_col, _y_col, _fname in _scatter_pairs:
                fig, ax = plt.subplots(figsize=(7, 5))
                for _i in range(CLF_K_FINAL):
                    _mask = pdf_fit_clf["CLF_CORRETOR"] == str(_i)
                    ax.scatter(
                        pdf_fit_clf.loc[_mask, _x_col],
                        pdf_fit_clf.loc[_mask, _y_col],
                        label=f"cluster {_i}",
                        alpha=0.5,
                        s=20,
                        color=_colors[_i % len(_colors)],
                    )
                ax.set_xlabel(_x_col)
                ax.set_ylabel(_y_col)
                ax.set_title(f"{_x_col} vs {_y_col} (K={CLF_K_FINAL})")
                ax.legend()
                _path = os.path.join(_tmpdir, _fname)
                fig.savefig(_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                mlflow.log_artifact(_path, artifact_path="clustering/original")

        # ── Visualizações em escala normalizada (clustering/scaled/) ─────────────
        with tempfile.TemporaryDirectory() as _tmpdir_scaled:
            _feat_idx = {feat: j for j, feat in enumerate(CLF_FEATURES)}

            # cluster_heatmap scaled
            _heatmap_scaled_df = pd.DataFrame(
                km.cluster_centers_,
                columns=CLF_FEATURES,
                index=[f"cluster_{i}" for i in range(CLF_K_FINAL)],
            )
            fig, ax = plt.subplots(figsize=(7, max(3, CLF_K_FINAL)))
            sns.heatmap(_heatmap_scaled_df, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                        ax=ax, linewidths=0.5)
            ax.set_title(f"Perfil de cluster — centroides em escala normalizada σ (K={CLF_K_FINAL})")
            _path = os.path.join(_tmpdir_scaled, "cluster_heatmap.png")
            fig.savefig(_path, bbox_inches="tight", dpi=120)
            plt.close(fig)
            mlflow.log_artifact(_path, artifact_path="clustering/scaled")

            # scatter plots scaled
            _scatter_pairs_scaled = [
                ("hr_mean",      "cotacao_mean", "scatter_hr_cotacao.png"),
                ("hr_mean",      "n_produtos",   "scatter_hr_produtos.png"),
                ("cotacao_mean", "n_produtos",   "scatter_cotacao_produtos.png"),
            ]
            for _x_col, _y_col, _fname in _scatter_pairs_scaled:
                _xi, _yi = _feat_idx[_x_col], _feat_idx[_y_col]
                fig, ax = plt.subplots(figsize=(7, 5))
                for _i in range(CLF_K_FINAL):
                    _mask = km.labels_ == _i
                    ax.scatter(
                        X_scaled[_mask, _xi],
                        X_scaled[_mask, _yi],
                        label=f"cluster {_i}",
                        alpha=0.5,
                        s=20,
                        color=_colors[_i % len(_colors)],
                    )
                ax.set_xlabel(f"{_x_col} (σ)")
                ax.set_ylabel(f"{_y_col} (σ)")
                ax.set_title(f"{_x_col} vs {_y_col} — escala normalizada (K={CLF_K_FINAL})")
                ax.legend()
                _path = os.path.join(_tmpdir_scaled, _fname)
                fig.savefig(_path, bbox_inches="tight", dpi=120)
                plt.close(fig)
                mlflow.log_artifact(_path, artifact_path="clustering/scaled")

        # ── Salvar tabelas gold ───────────────────────────────────────────────────
        df_model_out.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_MODEL_FQN)
        df_valid_out.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_VALID_FQN)
        mlflow.log_metrics({"df_model_saved": 1, "df_validacao_saved": 1})

print("✅ T_CLUSTERING_FIT ok")
print(f"• K_FINAL={CLF_K_FINAL} | silhouette={clf_silhouette:.4f} | inertia={clf_inertia:.1f}")
print(f"• cobertura model={pct_cob_model:.1%} | valid={pct_cob_valid:.1%}")
print("• df_model    :", DF_MODEL_FQN, f"({n_model_total} linhas)")
print("• df_validacao:", DF_VALID_FQN, f"({n_valid_total} linhas)")
for i in range(CLF_K_FINAL):
    print(f"  cluster {i}: {clf_cluster_dist[f'clf_cluster_{i}_n_corretores']} corretores")

# COMMAND ----------

spark.catalog.tableExists(DF_MODEL_FQN)

# COMMAND ----------

