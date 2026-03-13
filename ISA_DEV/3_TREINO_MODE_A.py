# Databricks notebook source
x = 1

# COMMAND ----------

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

PR_TREINO_NAME  = "T_PR_TREINO"   # container geral (fixo)
MODE_NAME       = "T_MODE_A"      # branch/abordagem (fixo)
MODE_CODE       = "A"

# =========================
# Acoplar a hierarquia existente (opcional)
# Preencha com os run_ids da hierarquia que deseja reutilizar.
# Deixe vazio ("") para criar uma nova hierarquia.
# =========================
PR_RUN_ID_OVERRIDE        = ""  # run_id do T_PR_TREINO
MODE_RUN_ID_OVERRIDE      = ""  # run_id do T_MODE_A
PRE_PROC_RUN_ID_OVERRIDE  = ""  # run_id do container T_PRE_PROC_MODEL
FS_RUN_ID_OVERRIDE        = ""  # run_id do container T_FEATURE_SELECTION
TREINO_RUN_ID_OVERRIDE    = ""  # run_id do container T_TREINO

# Steps dentro do MODE (fixos)
STEP_PRE_PROC_NAME          = "T_PRE_PROC_MODEL"
STEP_FEATURE_SELECTION_NAME = "T_FEATURE_SELECTION"
STEP_TREINO_NAME            = "T_TREINO"

# =========================
# Versionamento / timestamp
# =========================
TREINO_VERSAO = "V6"
TREINO_VERSAO_TABLE_SAFE = TREINO_VERSAO.replace(".", "_")

TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_UUID = uuid.uuid4().hex[:8]

# Versão de referência logada como tag em todas as exec runs
VERSAO_REF = TREINO_VERSAO

RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    """
    Nome com timestamp (sem versão):
    ex.: T_PRE_PROC_MODEL_20260306_123456
    """
    return f"{base}_{TS_EXEC}"


# =========================
# INPUT (manual): tabela segmentada (saída do 2_JOIN)
# =========================
COTACAO_SEG_FQN = "silver.cotacao_seg_20260307_101741"  # <<< AJUSTE MANUALMENTE

# =========================
# OUTPUT (gold): df_model e df_validacao versionados
# =========================
OUT_SCHEMA = "gold"
DF_MODEL_FQN = f"{OUT_SCHEMA}.cotacao_model_{TS_EXEC}_{RUN_UUID}"
DF_VALID_FQN = f"{OUT_SCHEMA}.cotacao_validacao_{TS_EXEC}_{RUN_UUID}"
WRITE_MODE = "overwrite"

# =========================
# T_PRE_PROC_MODEL — parâmetros das regras
# =========================
STATUS_TO_PERDIDA  = ["EM ANALISE", "EM NEGOCIAÇÃO"]
EXCLUDE_DATE_START = "2025-09-01"
EXCLUDE_DATE_END   = "2025-12-31"
ALLOWED_FINAL_STATUS = ["Emitida", "Perdida"]
LABEL_COL = "label"

ID_COL    = "CD_NUMERO_COTACAO_AXA"
SEG_COL   = "SEG"
DATE_COL  = "DATA_COTACAO"
VALID_FRAC = 0.20
SPLIT_SALT = "split_v2_seg_mes"

# =========================
# Profiling/EDA simples
# =========================
DO_PROFILE = True
STATUS_COL = "DS_GRUPO_STATUS"

# =========================
# FS e TREINO
# =========================
SEG_TARGET = "SEGURO_NOVO_MANUAL"   # <<< escolha manual para FS e treino
SEED = 42
TRAIN_FRAC = 0.80

FS_METHODS = ["lr_l1", "rf", "gbt"]
TOPK_LIST = [5, 7, 12]

TRAIN_MODELS = ["gbt"]
THRESHOLDS = [round(x, 2) for x in [i/100 for i in range(5, 96, 5)]]
PRECISION_TARGET = 0.20

ID_COLS = [ID_COL, "CD_DOC_CORRETOR", "TS_ARQ", SEG_COL, DATE_COL]
DROP_FROM_FEATURES = ID_COLS + [STATUS_COL]

DECIMAL_COLS = [
    "VL_PREMIO_ALVO", "VL_PREMIO_LIQUIDO", "VL_PRE_TOTAL", "VL_ENDOSSO_PREMIO_TOTAL",
    "VL_GWP_CORRETOR_RESUMO",
    "HR_2024_DETALHE", "HR_2025_DETALHE", "HR_M2_DETALHE", "HR_M3_DETALHE",
]

DIAS_COLS = [
    "DIAS_INICIO_VIGENCIA", "DIAS_VALIDADE", "DIAS_ANALISE_SUBSCRICAO",
    "DIAS_FIM_ANALISE_SUBSCRICAO", "DIAS_COTACAO", "DIAS_ULTIMA_ATUALIZACAO",
]

CAT_COLS = [
    "INTERMENDIARIO_PERFIL",
    "DS_PRODUTO_NOME",
    "DS_SISTEMA",
    "CD_FILIAL_RESPONSAVEL_COTACAO",
    "DS_ATIVIDADE_SEGURADO",
    "DS_GRUPO_CORRETOR_SEGMENTO",
]

QTD_INT_COLS = [
    "QTD_ACORDO_COMERCIAL_RESUMO",
    "QTD_COTACAO_2024_DETALHE", "QTD_COTACAO_2025_DETALHE", "QTD_COTACAO_M2_DETALHE", "QTD_COTACAO_M3_DETALHE",
    "QTD_EMITIDO_2024_DETALHE", "QTD_EMITIDO_2025_DETALHE", "QTD_EMITIDO_M2_DETALHE", "QTD_EMITIDO_M3_DETALHE",
]

print("✅ CONFIG TREINO carregada")
print("• input cotacao_seg:", COTACAO_SEG_FQN)
print("• outputs:", DF_MODEL_FQN, DF_VALID_FQN)
print("• SEG_TARGET (FS/TREINO):", SEG_TARGET)
print("• VERSAO_REF:", VERSAO_REF)
print("• RUN_SUFFIX:", RUN_SUFFIX)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define helpers MLflow

# COMMAND ----------

import json
from typing import Callable, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# =========================
# Metastore / MLflow helpers
# =========================
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


# =========================
# Regras dinâmicas (catálogo + executor)
# =========================
RuleFn = Callable[[DataFrame], DataFrame]

def rule_def(
    rule_id: str,
    description: str,
    fn: RuleFn,
    enabled: bool = True,
    requires_columns: Optional[List[str]] = None,
) -> Dict:
    return {
        "rule_id": rule_id,
        "description": description,
        "enabled": enabled,
        "requires_columns": requires_columns or [],
        "fn": fn,
    }

def safe_drop_cols(df: DataFrame, cols: List[str]) -> DataFrame:
    existing_cols = set(df.columns)
    cols_to_drop = [c for c in cols if c in existing_cols]
    if cols_to_drop:
        return df.drop(*cols_to_drop)
    return df

def apply_rules_block(
    block_key: str,
    df: DataFrame,
    rules: List[Dict],
    enable_rules: bool = True,
    toggles: Optional[Dict] = None,
) -> Tuple[DataFrame, List[Dict]]:
    """
    Aplica regras em ordem e retorna df_out e exec_log.
    Status: APPLIED | SKIPPED_DISABLED | SKIPPED_MISSING_COLS
    """
    toggles = toggles or {}
    df_out = df
    exec_log: List[Dict] = []

    for r in rules:
        rid = r["rule_id"]
        desc = r["description"]
        req = r.get("requires_columns", []) or []

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
            raise RuntimeError(f"❌ Falha regra {block_key}.{rid}: {desc}. Erro: {e}") from e

    return df_out, exec_log

def rules_catalog_for_logging(rules_by_block: Dict[str, List[Dict]]) -> Dict:
    out = {}
    for blk, rules in rules_by_block.items():
        out[blk] = [
            {
                "rule_id": r["rule_id"],
                "description": r["description"],
                "enabled": bool(r.get("enabled", True)),
                "requires_columns": r.get("requires_columns", []) or [],
            }
            for r in rules
        ]
    return out


# =========================
# Profiling simples
# =========================
def profile_basic(df: DataFrame, name: str, key_cols: Optional[List[str]] = None) -> Dict:
    n_rows = df.count()
    n_cols = len(df.columns)

    exprs = []
    for c, t in df.dtypes:
        if t == "string":
            exprs.append(F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1).otherwise(0)).alias(c))
        else:
            exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c))

    nulls = df.agg(*exprs).collect()[0].asDict()

    out = {
        "name": name,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "null_count": {k: int(v) for k, v in nulls.items()},
    }

    if key_cols:
        distincts = {}
        for k in key_cols:
            if k in df.columns:
                distincts[k] = int(df.select(k).distinct().count())
        out["distinct_count"] = distincts

    return out


# =========================
# EDA simples "por SEG"
# =========================
def counts_by_seg(df: DataFrame, seg_col: str) -> List[Dict]:
    if seg_col not in df.columns:
        return []
    rows = (df.groupBy(seg_col).count().orderBy(F.col("count").desc()).collect())
    return [{seg_col: r[seg_col], "count": int(r["count"])} for r in rows]

def status_dist_by_seg(df: DataFrame, seg_col: str, status_col: str) -> List[Dict]:
    if seg_col not in df.columns or status_col not in df.columns:
        return []
    base = df.groupBy(seg_col, status_col).count()
    tot = df.groupBy(seg_col).count().withColumnRenamed("count", "seg_total")
    out = (base.join(tot, on=seg_col, how="left")
               .withColumn("pct", (F.col("count") / F.col("seg_total")).cast("double"))
               .orderBy(F.col(seg_col), F.col("count").desc()))
    rows = out.collect()
    return [
        {seg_col: r[seg_col], status_col: r[status_col], "count": int(r["count"]), "seg_total": int(r["seg_total"]), "pct": float(r["pct"])}
        for r in rows
    ]

def label_rate_by_seg(df: DataFrame, seg_col: str, label_col: str) -> List[Dict]:
    if seg_col not in df.columns or label_col not in df.columns:
        return []
    agg = (df.groupBy(seg_col)
             .agg(
                 F.count(F.lit(1)).alias("n"),
                 F.avg(F.col(label_col).cast("double")).alias("label_rate"),
             )
             .orderBy(F.col("n").desc()))
    rows = agg.collect()
    return [{seg_col: r[seg_col], "n": int(r["n"]), "label_rate": float(r["label_rate"])} for r in rows]

def date_range_by_seg(df: DataFrame, seg_col: str, date_col: str) -> List[Dict]:
    if seg_col not in df.columns or date_col not in df.columns:
        return []
    d = F.to_date(F.col(date_col))
    agg = (df.withColumn("_d", d)
             .groupBy(seg_col)
             .agg(F.min("_d").alias("min_date"), F.max("_d").alias("max_date"), F.count(F.lit(1)).alias("n"))
             .orderBy(F.col("n").desc()))
    rows = agg.collect()
    out = []
    for r in rows:
        out.append({
            seg_col: r[seg_col],
            "n": int(r["n"]),
            "min_date": r["min_date"].isoformat() if r["min_date"] else None,
            "max_date": r["max_date"].isoformat() if r["max_date"] else None,
        })
    return out


# =========================
# Lineage helpers
# =========================
def build_tables_lineage_preproc() -> Dict:
    return {
        "stage": "T_PRE_PROC_MODEL",
        "ts_exec": TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode": MODE_CODE,
        "inputs": {"cotacao_seg": COTACAO_SEG_FQN},
        "outputs": {"df_model": DF_MODEL_FQN, "df_validacao": DF_VALID_FQN},
    }

def build_tables_lineage_fs_train(df_model_fqn: str) -> Dict:
    return {
        "ts_exec": TS_EXEC,
        "treino_versao": TREINO_VERSAO,
        "mode": MODE_CODE,
        "inputs": {"df_model": df_model_fqn},
        "seg_target": SEG_TARGET,
    }


print("✅ CEL 1 carregada: regras + profiling/EDA + lineage helpers (TREINO)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define estrutura MLflow

# COMMAND ----------

# =========================
# Validações + schema
# =========================
ensure_schema(OUT_SCHEMA)
assert_table_exists(COTACAO_SEG_FQN)

if table_exists(DF_MODEL_FQN):
    raise ValueError(f"❌ DF_MODEL já existe: {DF_MODEL_FQN} (timestamp/uuid repetido?)")
if table_exists(DF_VALID_FQN):
    raise ValueError(f"❌ DF_VALID já existe: {DF_VALID_FQN} (timestamp/uuid repetido?)")

# =========================
# MLflow experiment
# =========================
_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# Se houver run ativa (rerun), encerra para evitar stack inconsistente
while mlflow.active_run() is not None:
    mlflow.end_run()

# =========================
# Start (ou acopla) PR — container geral — NÃO LOGA NADA
# =========================
if PR_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    PR_RUN_ID = PR_RUN_ID_OVERRIDE
    _pr_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=PR_TREINO_NAME)
    PR_RUN_ID = mlflow.active_run().info.run_id
    _pr_status = "nova"

# =========================
# Start (ou acopla) MODE — container — NÃO LOGA NADA
# =========================
if MODE_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=MODE_RUN_ID_OVERRIDE, nested=True)
    MODE_RUN_ID = MODE_RUN_ID_OVERRIDE
    _mode_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=MODE_NAME, nested=True)
    MODE_RUN_ID = mlflow.active_run().info.run_id
    _mode_status = "nova"

print("✅ MLflow bootstrap ok (containers sem logging)")
print(f"• PR_RUN_ID  : {PR_RUN_ID} | {PR_TREINO_NAME} [{_pr_status}]")
print(f"• MODE_RUN_ID: {MODE_RUN_ID} | {MODE_NAME} [{_mode_status}]")
print(f"• Active run : {mlflow.active_run().info.run_id} (MODE)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_PRE_PROC_MODEL

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funções das regras — T_PRE_PROC_MODEL

# COMMAND ----------

# =========================================================
# Funções de regra — T_PRE_PROC_MODEL
# =========================================================

def PP_R01_normaliza_status_emitida_perdida(df: DataFrame) -> DataFrame:
    s = F.upper(F.trim(F.col(STATUS_COL).cast("string")))
    return df.withColumn(
        STATUS_COL,
        F.when(s == "EMITIDA", F.lit("Emitida"))
         .when(s == "PERDIDA", F.lit("Perdida"))
         .otherwise(F.col(STATUS_COL))
    )

def PP_R02_status_para_perdida(df: DataFrame) -> DataFrame:
    s = F.upper(F.trim(F.col(STATUS_COL).cast("string")))
    alvo = {"EM ANALISE", "EM ANÁLISE", "EM NEGOCIAÇÃO", "EM NEGOCIACAO"}
    return df.withColumn(STATUS_COL, F.when(s.isin(list(alvo)), F.lit("Perdida")).otherwise(F.col(STATUS_COL)))

def PP_R03_exclui_janela(df: DataFrame) -> DataFrame:
    d = F.to_date(F.col(DATE_COL))
    start = F.to_date(F.lit(EXCLUDE_DATE_START))
    end = F.to_date(F.lit(EXCLUDE_DATE_END))
    return df.withColumn("_d", d).filter(~(F.col("_d").between(start, end))).drop("_d")

def PP_R04_filtra_status_finais(df: DataFrame) -> DataFrame:
    return df.filter(F.col(STATUS_COL).isin(ALLOWED_FINAL_STATUS))

def PP_R05_cria_label(df: DataFrame) -> DataFrame:
    return df.withColumn(
        LABEL_COL,
        F.when(F.col(STATUS_COL) == "Emitida", F.lit(1.0))
         .when(F.col(STATUS_COL) == "Perdida", F.lit(0.0))
         .otherwise(F.lit(None).cast("double"))
    )

def BUILD_R01_add_mes(df: DataFrame) -> DataFrame:
    return (df.withColumn("DATA_COTACAO_dt", F.to_date(F.col(DATE_COL)))
              .filter(F.col("DATA_COTACAO_dt").isNotNull())
              .withColumn("MES", F.date_format(F.col("DATA_COTACAO_dt"), "yyyy-MM")))

def BUILD_R02_add_split_flag(df: DataFrame) -> DataFrame:
    h = F.xxhash64(
        F.col(ID_COL).cast("string"),
        F.col(SEG_COL).cast("string"),
        F.col("MES").cast("string"),
        F.lit(SPLIT_SALT),
    )
    score = (F.pmod(F.abs(h), F.lit(1000000)) / F.lit(1000000.0))
    return df.withColumn("_split_score", score).withColumn("is_valid", (F.col("_split_score") < F.lit(float(VALID_FRAC))))

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

# =========================================================
# Toggles de regras — RULES_ON_DF_SEG
# =========================================================
# True = regra ativa | False = regra desabilitada
TOGGLES_RULES_ON_DF_SEG = {
    "PP_R01": True,   # Normalizar DS_GRUPO_STATUS finais (EMITIDA->Emitida, PERDIDA->Perdida)
    "PP_R02": True,   # Setar DS_GRUPO_STATUS='Perdida' para status em análise/negociação
    "PP_R03": False,  # Excluir DATA_COTACAO entre EXCLUDE_DATE_START e EXCLUDE_DATE_END (desabilitado: reexecução V6)
    "PP_R04": True,   # Manter apenas status finais (Emitida, Perdida)
    "PP_R05": True,   # Criar coluna label (Emitida=1.0, Perdida=0.0)
}

# =========================================================
# Toggles de regras — RULES_BUILD_BASE
# =========================================================
TOGGLES_RULES_BUILD_BASE = {
    "BUILD_R01": True,  # Criar MES=yyyy-MM a partir de DATA_COTACAO (filtra DATA_COTACAO nula)
    "BUILD_R02": True,  # Criar split determinístico por (ID, SEG, MES, salt) -> is_valid
}

# =========================================================
# Toggles de regras — RULES_BUILD_MODEL
# =========================================================
TOGGLES_RULES_BUILD_MODEL = {
    "MODEL_R03": True,  # Selecionar df_model (is_valid=False)
    "BUILD_R04": True,  # Remover colunas auxiliares do split
}

# =========================================================
# Toggles de regras — RULES_BUILD_VALID
# =========================================================
TOGGLES_RULES_BUILD_VALID = {
    "VALID_R03": True,  # Selecionar df_validacao (is_valid=True)
    "BUILD_R04": True,  # Remover colunas auxiliares do split
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Catálogo de regras — T_PRE_PROC_MODEL

# COMMAND ----------

# =========================================================
# Catálogo dinâmico (RULES_BY_BLOCK)
# =========================================================
RULES_ON_DF_SEG = [
    rule_def("PP_R01", "Normalizar DS_GRUPO_STATUS finais (EMITIDA->Emitida, PERDIDA->Perdida)",
             PP_R01_normaliza_status_emitida_perdida,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R01"],
             requires_columns=[STATUS_COL]),
    rule_def("PP_R02", "Setar DS_GRUPO_STATUS='Perdida' para status em análise/negociação",
             PP_R02_status_para_perdida,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R02"],
             requires_columns=[STATUS_COL]),
    rule_def("PP_R03", f"Excluir {DATE_COL} entre {EXCLUDE_DATE_START} e {EXCLUDE_DATE_END}",
             PP_R03_exclui_janela,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R03"],
             requires_columns=[DATE_COL]),
    rule_def("PP_R04", f"Manter apenas status finais {ALLOWED_FINAL_STATUS}",
             PP_R04_filtra_status_finais,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R04"],
             requires_columns=[STATUS_COL]),
    rule_def("PP_R05", "Criar label (Emitida=1.0, Perdida=0.0)",
             PP_R05_cria_label,
             enabled=TOGGLES_RULES_ON_DF_SEG["PP_R05"],
             requires_columns=[STATUS_COL]),
]

RULES_BUILD_BASE = [
    rule_def("BUILD_R01", "Criar MES=yyyy-MM a partir de DATA_COTACAO (filtra DATA_COTACAO nula)",
             BUILD_R01_add_mes,
             enabled=TOGGLES_RULES_BUILD_BASE["BUILD_R01"],
             requires_columns=[DATE_COL]),
    rule_def("BUILD_R02", "Criar split determinístico por (ID, SEG, MES, salt) -> is_valid",
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
    "rules_on_df_seg":       RULES_ON_DF_SEG,
    "rules_build_base":      RULES_BUILD_BASE,
    "rules_build_df_model":  RULES_BUILD_MODEL,
    "rules_build_df_validacao": RULES_BUILD_VALID,
}

print("✅ RULES_BY_BLOCK definido")
print("• rules_on_df_seg:", len(RULES_ON_DF_SEG))
print("• rules_build_base:", len(RULES_BUILD_BASE))
print("• rules_build_df_model:", len(RULES_BUILD_MODEL))
print("• rules_build_df_validacao:", len(RULES_BUILD_VALID))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execução — T_PRE_PROC_MODEL

# COMMAND ----------

# =========================
# T_PRE_PROC_MODEL — CONTAINER (sem logging) + execução versionada (com logging)
# =========================
RUN_PRE_PROC_EXEC = run_name_vts("T_PRE_PROC_MODEL")

_preproc_kw = {"run_id": PRE_PROC_RUN_ID_OVERRIDE} if PRE_PROC_RUN_ID_OVERRIDE else {"run_name": STEP_PRE_PROC_NAME}
with mlflow.start_run(**_preproc_kw, nested=True) as preproc_container:
    PREPROC_CONTAINER_RUN_ID = preproc_container.info.run_id

    with mlflow.start_run(run_name=RUN_PRE_PROC_EXEC, nested=True) as run:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("stage", "TREINO")
        mlflow.set_tag("run_role", "exec")
        mlflow.set_tag("mode", MODE_CODE)
        mlflow.set_tag("step", "PRE_PROC_MODEL")
        mlflow.set_tag("treino_versao", TREINO_VERSAO)
        mlflow.set_tag("versao_ref", VERSAO_REF)
        mlflow.set_tag("seg_target", SEG_TARGET)

        # params "globais"
        mlflow.log_param("ts_exec", TS_EXEC)
        mlflow.log_param("treino_versao", TREINO_VERSAO)
        mlflow.log_param("mode_code", MODE_CODE)
        mlflow.log_param("seg_target", SEG_TARGET)

        # lineage + outputs
        mlflow.log_param("input_cotacao_seg_fqn", COTACAO_SEG_FQN)
        mlflow.log_param("df_model_fqn", DF_MODEL_FQN)
        mlflow.log_param("df_validacao_fqn", DF_VALID_FQN)
        mlflow.log_param("run_suffix", RUN_SUFFIX)

        # ids dos containers (para rastreabilidade)
        mlflow.log_param("pr_run_id", PR_RUN_ID)
        mlflow.log_param("mode_run_id", MODE_RUN_ID)
        mlflow.log_param("t_pre_proc_model_container_run_id", PREPROC_CONTAINER_RUN_ID)

        # artifact run_tree
        run_tree = {
            "experiment": EXPERIMENT_NAME,
            "ts_exec": TS_EXEC,
            "treino_versao": TREINO_VERSAO,
            "mode": MODE_CODE,
            "containers": {
                "T_PR_TREINO": {"run_id": PR_RUN_ID, "run_name": PR_TREINO_NAME},
                "T_MODE_A": {"run_id": MODE_RUN_ID, "run_name": MODE_NAME},
                "T_PRE_PROC_MODEL": {"run_id": PREPROC_CONTAINER_RUN_ID, "run_name": STEP_PRE_PROC_NAME},
            },
            "exec": {"run_id": run.info.run_id, "run_name": RUN_PRE_PROC_EXEC},
            "inputs": {"cotacao_seg": COTACAO_SEG_FQN},
            "outputs": {"df_model": DF_MODEL_FQN, "df_validacao": DF_VALID_FQN},
        }
        mlflow.log_dict(run_tree, "run_tree.json")

        # params da etapa
        mlflow.log_param("exclude_date_start", EXCLUDE_DATE_START)
        mlflow.log_param("exclude_date_end", EXCLUDE_DATE_END)
        mlflow.log_param("allowed_final_status", json.dumps(ALLOWED_FINAL_STATUS, ensure_ascii=False))
        mlflow.log_param("status_to_perdida", json.dumps(STATUS_TO_PERDIDA, ensure_ascii=False))
        mlflow.log_param("valid_frac", float(VALID_FRAC))
        mlflow.log_param("split_salt", SPLIT_SALT)
        mlflow.log_param("id_col", ID_COL)
        mlflow.log_param("seg_col", SEG_COL)
        mlflow.log_param("date_col", DATE_COL)
        mlflow.log_param("label_col", LABEL_COL)
        mlflow.log_param("write_mode", WRITE_MODE)

        # toggles como params para rastreabilidade
        mlflow.log_dict(TOGGLES_RULES_ON_DF_SEG, "toggles/rules_on_df_seg.json")
        mlflow.log_dict(TOGGLES_RULES_BUILD_BASE, "toggles/rules_build_base.json")
        mlflow.log_dict(TOGGLES_RULES_BUILD_MODEL, "toggles/rules_build_df_model.json")
        mlflow.log_dict(TOGGLES_RULES_BUILD_VALID, "toggles/rules_build_df_validacao.json")

        # lineage do step
        mlflow.log_dict(build_tables_lineage_preproc(), "tables_lineage.json")

        # catálogo de regras
        mlflow.log_dict(rules_catalog_for_logging(RULES_BY_BLOCK), "rules_catalog.json")

        # -------------------------
        # Carrega df_seg (input)
        # -------------------------
        df_seg_in = spark.table(COTACAO_SEG_FQN)
        n_seg_in = df_seg_in.count()
        mlflow.log_metric("n_seg_in", int(n_seg_in))

        # -------------------------
        # Aplica regras
        # -------------------------
        exec_log = {}

        df_seg_pp, exec_log["rules_on_df_seg"] = apply_rules_block(
            "rules_on_df_seg", df_seg_in, RULES_ON_DF_SEG, True
        )
        mlflow.log_metric("n_seg_after_rules", int(df_seg_pp.count()))

        df_base, exec_log["rules_build_base"] = apply_rules_block(
            "rules_build_base", df_seg_pp, RULES_BUILD_BASE, True
        )
        df_model_tmp, exec_log["rules_build_df_model"] = apply_rules_block(
            "rules_build_df_model", df_base, RULES_BUILD_MODEL, True
        )
        df_valid_tmp, exec_log["rules_build_df_validacao"] = apply_rules_block(
            "rules_build_df_validacao", df_base, RULES_BUILD_VALID, True
        )

        mlflow.log_dict(exec_log, "rules_execution.json")

        n_model = df_model_tmp.count()
        n_valid = df_valid_tmp.count()
        mlflow.log_metric("n_df_model", int(n_model))
        mlflow.log_metric("n_df_validacao", int(n_valid))

        if DO_PROFILE:
            mlflow.log_dict(profile_basic(df_model_tmp, name="df_model", key_cols=[ID_COL, SEG_COL]), "profiling_df_model.json")
            mlflow.log_dict(profile_basic(df_valid_tmp, name="df_validacao", key_cols=[ID_COL, SEG_COL]), "profiling_df_validacao.json")

            eda_model = {
                "counts_by_seg": counts_by_seg(df_model_tmp, SEG_COL),
                "status_dist_by_seg": status_dist_by_seg(df_model_tmp, SEG_COL, STATUS_COL),
                "label_rate_by_seg": label_rate_by_seg(df_model_tmp, SEG_COL, LABEL_COL),
                "date_range_by_seg": date_range_by_seg(df_model_tmp, SEG_COL, DATE_COL),
            }
            eda_valid = {
                "counts_by_seg": counts_by_seg(df_valid_tmp, SEG_COL),
                "status_dist_by_seg": status_dist_by_seg(df_valid_tmp, SEG_COL, STATUS_COL),
                "label_rate_by_seg": label_rate_by_seg(df_valid_tmp, SEG_COL, LABEL_COL),
                "date_range_by_seg": date_range_by_seg(df_valid_tmp, SEG_COL, DATE_COL),
            }
            mlflow.log_dict(eda_model, "eda_df_model_by_seg.json")
            mlflow.log_dict(eda_valid, "eda_df_validacao_by_seg.json")

        # salva só no final
        df_model_tmp.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_MODEL_FQN)
        df_valid_tmp.write.format("delta").mode(WRITE_MODE).saveAsTable(DF_VALID_FQN)
        mlflow.log_metric("df_model_saved", 1)
        mlflow.log_metric("df_validacao_saved", 1)

print("✅ T_PRE_PROC_MODEL ok (container limpo; logging na execução versionada)")
print("• exec run:", RUN_PRE_PROC_EXEC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_FEATURE_SELECTION

# COMMAND ----------

import os
import json
import tempfile
import pandas as pd

import mlflow
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# =========================
# Helpers locais — FS
# =========================
def compute_logloss(df_pred, label_col="label", prob_col="probability", eps=1e-15):
    p1 = vector_to_array(F.col(prob_col)).getItem(1)
    p1c = F.least(F.greatest(p1, F.lit(eps)), F.lit(1.0 - eps))
    y = F.col(label_col).cast("double")
    ll = df_pred.select(
        (- (y * F.log(p1c) + (1.0 - y) * F.log(1.0 - p1c))).alias("ll")
    ).agg(F.avg("ll").alias("logloss")).collect()[0]["logloss"]
    return float(ll)

def get_vector_attr_names(df_with_vec, vec_col="features_vec"):
    meta = df_with_vec.schema[vec_col].metadata
    if "ml_attr" not in meta:
        raise ValueError(f"Coluna {vec_col} não tem metadata ml_attr.")
    ml_attr = meta["ml_attr"]
    attrs = []
    if "attrs" in ml_attr:
        for k in ["binary", "numeric", "nominal"]:
            if k in ml_attr["attrs"]:
                attrs.extend(ml_attr["attrs"][k])
    attrs_sorted = sorted(attrs, key=lambda x: x["idx"])
    return [a["name"] for a in attrs_sorted]

def base_feature_from_attr(attr_name: str):
    if "=" in attr_name:
        return attr_name.split("=", 1)[0]
    if "__ohe" in attr_name:
        return attr_name.split("__ohe", 1)[0]
    if "__imp" in attr_name:
        return attr_name.split("__imp", 1)[0]
    return attr_name

def importance_to_tables(model_key: str, importances, attr_names):
    pdf_raw = pd.DataFrame({"attr": attr_names, "importance": importances})
    pdf_raw["feature"] = pdf_raw["attr"].apply(base_feature_from_attr)

    pdf_agg = pdf_raw.groupby("feature", as_index=False)["importance"].sum()
    pdf_agg = pdf_agg.sort_values("importance", ascending=False).reset_index(drop=True)

    n = len(pdf_agg)
    if n <= 1:
        pdf_agg["rank"] = 1
        pdf_agg["score"] = 1.0
    else:
        pdf_agg["rank"] = (pdf_agg["importance"].rank(method="first", ascending=False)).astype(int)
        pdf_agg["score"] = 1.0 - (pdf_agg["rank"] - 1) / (n - 1)

    pdf_scored = pdf_agg.rename(columns={
        "importance": f"importance_{model_key}",
        "rank": f"rank_{model_key}",
        "score": f"score_{model_key}",
    })
    return pdf_raw, pdf_scored

def log_pandas_csv(pdf: pd.DataFrame, artifact_path: str):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        pdf.to_csv(fp, index=False)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

def cast_numeric_to_double(df, cols):
    out = df
    for c in cols:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast("double"))
    return out

def clean_str(colname: str):
    x = F.trim(F.col(colname).cast("string"))
    x = (F.when(F.col(colname).isNull(), F.lit(None))
           .when(F.length(x) == 0, F.lit(None))
           .when(F.upper(x).isin(["NULL", "N/A", "NA", "NONE", "-"]), F.lit(None))
           .otherwise(x))
    return x


# =========================
# FS config
# =========================
NULL_DROP_PCT = 0.90
HIGH_CARD_THRESHOLD = 300

RUN_FS_EXEC = run_name_vts("T_FS")

# ordem determinística
ordered_methods = [m for m in ["lr_l1", "rf", "gbt"] if m in FS_METHODS]
if not ordered_methods:
    raise ValueError("❌ FS_METHODS vazio. Defina ao menos um método.")


# =========================
# T_FEATURE_SELECTION — CONTAINER (sem logging)
# =========================
_fs_kw = {"run_id": FS_RUN_ID_OVERRIDE} if FS_RUN_ID_OVERRIDE else {"run_name": STEP_FEATURE_SELECTION_NAME}
with mlflow.start_run(**_fs_kw, nested=True) as fs_container:
    FS_CONTAINER_RUN_ID = fs_container.info.run_id

    with mlflow.start_run(run_name=RUN_FS_EXEC, nested=True) as r:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("stage", "TREINO")
        mlflow.set_tag("run_role", "exec")
        mlflow.set_tag("mode", MODE_CODE)
        mlflow.set_tag("step", "FEATURE_SELECTION")
        mlflow.set_tag("treino_versao", TREINO_VERSAO)
        mlflow.set_tag("versao_ref", VERSAO_REF)

        # params principais
        mlflow.log_param("df_model_fqn", DF_MODEL_FQN)
        mlflow.log_param("seg_target", SEG_TARGET)
        mlflow.log_param("seed", int(SEED))
        mlflow.log_param("train_frac_internal", float(TRAIN_FRAC))
        mlflow.log_param("null_drop_pct", float(NULL_DROP_PCT))
        mlflow.log_param("high_card_threshold", int(HIGH_CARD_THRESHOLD))
        mlflow.log_param("fs_methods", json.dumps(ordered_methods))
        mlflow.log_param("topk_list", json.dumps(TOPK_LIST))
        mlflow.log_param("run_suffix", RUN_SUFFIX)

        mlflow.log_dict(
            {
                "stage": "T_FEATURE_SELECTION",
                "mode": MODE_CODE,
                "seg_target": SEG_TARGET,
                "input_df_model": DF_MODEL_FQN,
                "run_suffix": RUN_SUFFIX,
            },
            "tables_lineage.json",
        )

        # -------------------------
        # Load + filter SEG_TARGET
        # -------------------------
        df_raw = spark.table(DF_MODEL_FQN)
        df_seg = df_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()

        n_rows = df_seg.count()
        n_ids  = df_seg.select(ID_COL).distinct().count()

        df_lab = df_seg.withColumn("label_int", F.col(LABEL_COL).cast("int"))
        n_label_invalid = df_lab.filter(~F.col("label_int").isin([0, 1]) | F.col("label_int").isNull()).count()
        df_base = df_lab.drop(LABEL_COL).withColumnRenamed("label_int", "label")

        for c, t in df_base.dtypes:
            if t == "string":
                df_base = df_base.withColumn(c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c)))

        n_dups = df_base.groupBy(ID_COL).count().filter(F.col("count") > 1).count()

        mlflow.log_metrics({
            "n_rows_seg": int(n_rows),
            "n_ids_seg": int(n_ids),
            "n_label_invalid_or_null": int(n_label_invalid),
            "n_ids_with_dups": int(n_dups),
        })

        # -------------------------
        # Perfil null/blank em candidatas
        # -------------------------
        cols_candidate = list(dict.fromkeys(ID_COLS + ["label"] + DECIMAL_COLS + DIAS_COLS + QTD_INT_COLS + CAT_COLS))
        cols_candidate = [c for c in cols_candidate if c in df_base.columns]

        exprs = []
        for c, t in df_base.select(*cols_candidate).dtypes:
            exprs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__nulls"))
            if t == "string":
                exprs.append(F.sum(F.when(F.length(F.trim(F.col(c))) == 0, 1).otherwise(0)).alias(f"{c}__blanks"))
        agg_row = df_base.select(*cols_candidate).agg(*exprs).collect()[0].asDict()

        profile_rows = []
        for c, t in df_base.select(*cols_candidate).dtypes:
            nulls  = int(agg_row.get(f"{c}__nulls", 0))
            blanks = int(agg_row.get(f"{c}__blanks", 0)) if t == "string" else 0
            profile_rows.append({"col": c, "dtype": t, "nulls": nulls, "blanks": blanks, "pct_null": (nulls / n_rows if n_rows else None)})

        mlflow.log_dict({"null_blank_profile": profile_rows}, "fs_stage1/null_blank_profile.json")

        # -------------------------
        # Cardinalidade categóricas
        # -------------------------
        cat_present = [c for c in CAT_COLS if c in df_base.columns]
        cat_card = []
        for c in cat_present:
            cd = int(df_base.select(F.countDistinct(F.col(c)).alias("cd")).collect()[0]["cd"])
            cat_card.append({"col": c, "count_distinct": cd})
        high_card_cols = [r["col"] for r in cat_card if r["count_distinct"] > HIGH_CARD_THRESHOLD]

        mlflow.log_dict({"cat_cardinality": cat_card}, "fs_stage1/cat_cardinality.json")
        mlflow.log_dict({"high_card_cols": high_card_cols}, "fs_stage1/high_card_cols.json")

        # -------------------------
        # Cast QTD_* para int
        # -------------------------
        qtd_present = [c for c in QTD_INT_COLS if c in df_base.columns]
        df_cast = df_base
        for c in qtd_present:
            cleaned = clean_str(c)
            df_cast = df_cast.withColumn(
                c,
                F.when(cleaned.rlike(r"^-?\d+$"), cleaned.cast("int"))
                 .when(cleaned.rlike(r"^-?\d+(\.\d+)?$"), cleaned.cast("double").cast("int"))
                 .otherwise(F.lit(None).cast("int"))
            )
        df_base2 = df_cast

        # -------------------------
        # Drops + listas finais
        # -------------------------
        drop_null_cols = [r["col"] for r in profile_rows if (r["pct_null"] is not None and r["pct_null"] > NULL_DROP_PCT)]
        drop_constant_cols = [r["col"] for r in cat_card if r["count_distinct"] <= 1]
        DROP_FEATURES_FINAL = set(DROP_FROM_FEATURES + drop_null_cols + drop_constant_cols)

        NUM_COLS_FINAL = [c for c in (DECIMAL_COLS + DIAS_COLS + QTD_INT_COLS)
                          if c in df_base2.columns and c not in DROP_FEATURES_FINAL and c != "label"]
        CAT_COLS_FINAL = [c for c in CAT_COLS
                          if c in df_base2.columns and c not in DROP_FEATURES_FINAL and c != "label"]
        FEATURE_COLS_ALL = CAT_COLS_FINAL + NUM_COLS_FINAL

        if len(FEATURE_COLS_ALL) == 0:
            raise ValueError("❌ FEATURE_COLS_ALL vazio após drops.")

        fs_contract = {
            "seg_target": SEG_TARGET,
            "df_model_fqn": DF_MODEL_FQN,
            "null_drop_pct": NULL_DROP_PCT,
            "high_card_threshold": HIGH_CARD_THRESHOLD,
            "drop_null_cols": drop_null_cols,
            "drop_constant_cols": drop_constant_cols,
            "drop_features_final": sorted(list(DROP_FEATURES_FINAL)),
            "cat_cols_final": CAT_COLS_FINAL,
            "num_cols_final": NUM_COLS_FINAL,
            "features_total": int(len(FEATURE_COLS_ALL)),
            "high_card_cols": high_card_cols,
        }
        mlflow.log_dict(fs_contract, "fs_stage1/fs_feature_contract.json")

        # -------------------------
        # Split estratificado interno
        # -------------------------
        df_audit = df_base2.select(*[c for c in ID_COLS if c in df_base2.columns], "label", *FEATURE_COLS_ALL).cache()
        fractions = {0: TRAIN_FRAC, 1: TRAIN_FRAC}
        df_train_audit = df_audit.sampleBy("label", fractions=fractions, seed=SEED).cache()
        train_ids = df_train_audit.select(ID_COL).cache()
        df_val_audit = df_audit.join(train_ids, on=ID_COL, how="left_anti").cache()

        n_tr = int(df_train_audit.count())
        n_va = int(df_val_audit.count())
        mlflow.log_metrics({"n_train_internal": n_tr, "n_val_internal": n_va})

        df_train_ml = df_train_audit.select("label", *FEATURE_COLS_ALL)
        df_val_ml   = df_val_audit.select("label", *FEATURE_COLS_ALL)

        # -------------------------
        # Pipeline preprocess (fit 1x)
        # -------------------------
        df_train_fs = cast_numeric_to_double(df_train_ml, NUM_COLS_FINAL).cache()
        df_val_fs   = cast_numeric_to_double(df_val_ml,   NUM_COLS_FINAL).cache()

        df_train_fs_clean = df_train_fs
        df_val_fs_clean = df_val_fs
        for c in CAT_COLS_FINAL:
            df_train_fs_clean = df_train_fs_clean.withColumn(
                c, F.when((F.col(c).isNull()) | (F.trim(F.col(c)) == ""), F.lit(None)).otherwise(F.col(c))
            )
            df_val_fs_clean = df_val_fs_clean.withColumn(
                c, F.when((F.col(c).isNull()) | (F.trim(F.col(c)) == ""), F.lit(None)).otherwise(F.col(c))
            )

        idx_cols = [f"{c}__idx" for c in CAT_COLS_FINAL]
        ohe_cols = [f"{c}__ohe" for c in CAT_COLS_FINAL]
        imp_cols = [f"{c}__imp" for c in NUM_COLS_FINAL]

        stages = []
        for c, out_c in zip(CAT_COLS_FINAL, idx_cols):
            stages.append(StringIndexer(inputCol=c, outputCol=out_c, handleInvalid="keep"))
        if idx_cols:
            stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, dropLast=False))
        if NUM_COLS_FINAL:
            stages.append(Imputer(inputCols=NUM_COLS_FINAL, outputCols=imp_cols, strategy="mean"))
        stages.append(VectorAssembler(inputCols=ohe_cols + imp_cols, outputCol="features_vec"))

        preprocess = Pipeline(stages=stages).fit(df_train_fs_clean)
        df_train_vec = preprocess.transform(df_train_fs_clean).select("label", "features_vec").cache()
        df_val_vec   = preprocess.transform(df_val_fs_clean).select("label", "features_vec").cache()

        attr_names = get_vector_attr_names(df_train_vec, "features_vec")
        features_vec_dim = int(len(attr_names))
        mlflow.log_metric("features_vec_dim", features_vec_dim)

        evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        evaluator_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

        # =========================
        # Rodar métodos e logar tudo na MESMA run
        # =========================
        fs_tables = {}
        method_info = {}

        for m in ordered_methods:
            if m == "lr_l1":
                algo = "lr_l1"
                lr = LogisticRegression(featuresCol="features_vec", labelCol="label",
                                        maxIter=100, regParam=0.01, elasticNetParam=1.0, standardization=True)
                lr_model = lr.fit(df_train_vec)
                pred_val = lr_model.transform(df_val_vec)

                auc = float(evaluator_auc.evaluate(pred_val))
                prauc = float(evaluator_pr.evaluate(pred_val))
                logloss = float(compute_logloss(pred_val))
                mlflow.log_metrics({f"{algo}_auc_val": auc, f"{algo}_prauc_val": prauc, f"{algo}_logloss_val": logloss})

                coef = lr_model.coefficients.toArray()
                imp = [abs(float(x)) for x in coef]
                pdf_raw, pdf_scored = importance_to_tables("lr", imp, attr_names)

                all_feat = pd.DataFrame({"feature": FEATURE_COLS_ALL})
                pdf_scored = all_feat.merge(pdf_scored, on="feature", how="left").fillna({
                    "importance_lr": 0.0, "rank_lr": len(FEATURE_COLS_ALL), "score_lr": 0.0
                })

                log_pandas_csv(pdf_raw, f"methods/{algo}/importance_raw.csv")
                log_pandas_csv(pdf_scored.sort_values("score_lr", ascending=False), f"methods/{algo}/importance_by_feature.csv")

                ranked = pdf_scored.sort_values("score_lr", ascending=False)["feature"].tolist()
                method_sets = {f"top_{k}": ranked[:k] for k in TOPK_LIST}
                method_sets["all"] = ranked
                mlflow.log_dict(method_sets, f"methods/{algo}/topk_sets_method.json")

                fs_tables["lr"] = pdf_scored
                method_info[algo] = {"auc": auc, "prauc": prauc, "logloss": logloss}

            elif m == "rf":
                algo = "rf"
                rf = RandomForestClassifier(featuresCol="features_vec", labelCol="label",
                                            numTrees=200, maxDepth=8, featureSubsetStrategy="auto", seed=SEED)
                rf_model = rf.fit(df_train_vec)
                pred_val = rf_model.transform(df_val_vec)

                auc = float(evaluator_auc.evaluate(pred_val))
                prauc = float(evaluator_pr.evaluate(pred_val))
                logloss = float(compute_logloss(pred_val))
                mlflow.log_metrics({f"{algo}_auc_val": auc, f"{algo}_prauc_val": prauc, f"{algo}_logloss_val": logloss})

                imp = [float(x) for x in rf_model.featureImportances.toArray()]
                pdf_raw, pdf_scored = importance_to_tables("rf", imp, attr_names)

                all_feat = pd.DataFrame({"feature": FEATURE_COLS_ALL})
                pdf_scored = all_feat.merge(pdf_scored, on="feature", how="left").fillna({
                    "importance_rf": 0.0, "rank_rf": len(FEATURE_COLS_ALL), "score_rf": 0.0
                })

                log_pandas_csv(pdf_raw, f"methods/{algo}/importance_raw.csv")
                log_pandas_csv(pdf_scored.sort_values("score_rf", ascending=False), f"methods/{algo}/importance_by_feature.csv")

                ranked = pdf_scored.sort_values("score_rf", ascending=False)["feature"].tolist()
                method_sets = {f"top_{k}": ranked[:k] for k in TOPK_LIST}
                method_sets["all"] = ranked
                mlflow.log_dict(method_sets, f"methods/{algo}/topk_sets_method.json")

                fs_tables["rf"] = pdf_scored
                method_info[algo] = {"auc": auc, "prauc": prauc, "logloss": logloss}

            elif m == "gbt":
                algo = "gbt"
                gbt = GBTClassifier(featuresCol="features_vec", labelCol="label",
                                    maxIter=80, maxDepth=5, stepSize=0.1, seed=SEED)
                gbt_model = gbt.fit(df_train_vec)
                pred_val = gbt_model.transform(df_val_vec)

                auc = float(evaluator_auc.evaluate(pred_val))
                prauc = float(evaluator_pr.evaluate(pred_val))
                logloss = float(compute_logloss(pred_val))
                mlflow.log_metrics({f"{algo}_auc_val": auc, f"{algo}_prauc_val": prauc, f"{algo}_logloss_val": logloss})

                imp = [float(x) for x in gbt_model.featureImportances.toArray()]
                pdf_raw, pdf_scored = importance_to_tables("gbt", imp, attr_names)

                all_feat = pd.DataFrame({"feature": FEATURE_COLS_ALL})
                pdf_scored = all_feat.merge(pdf_scored, on="feature", how="left").fillna({
                    "importance_gbt": 0.0, "rank_gbt": len(FEATURE_COLS_ALL), "score_gbt": 0.0
                })

                log_pandas_csv(pdf_raw, f"methods/{algo}/importance_raw.csv")
                log_pandas_csv(pdf_scored.sort_values("score_gbt", ascending=False), f"methods/{algo}/importance_by_feature.csv")

                ranked = pdf_scored.sort_values("score_gbt", ascending=False)["feature"].tolist()
                method_sets = {f"top_{k}": ranked[:k] for k in TOPK_LIST}
                method_sets["all"] = ranked
                mlflow.log_dict(method_sets, f"methods/{algo}/topk_sets_method.json")

                fs_tables["gbt"] = pdf_scored
                method_info[algo] = {"auc": auc, "prauc": prauc, "logloss": logloss}

            else:
                raise ValueError(f"Método não suportado: {m}")

        # =========================
        # SUMMARY final
        # =========================
        pdf_final = pd.DataFrame({"feature": FEATURE_COLS_ALL})
        for key in ["lr", "rf", "gbt"]:
            if key in fs_tables:
                cols_keep = ["feature"] + [c for c in fs_tables[key].columns if c.startswith("score_") or c.startswith("rank_") or c.startswith("importance_")]
                pdf_final = pdf_final.merge(fs_tables[key][cols_keep], on="feature", how="left")

        for c in pdf_final.columns:
            if c.startswith("score_") or c.startswith("importance_"):
                pdf_final[c] = pdf_final[c].fillna(0.0)
            if c.startswith("rank_"):
                pdf_final[c] = pdf_final[c].fillna(len(FEATURE_COLS_ALL)).astype(int)

        score_cols = [c for c in pdf_final.columns if c.startswith("score_")]
        if not score_cols:
            raise ValueError("❌ Não há score_* para consolidar o summary.")

        pdf_final["score_final"] = pdf_final[score_cols].mean(axis=1)
        pdf_final = pdf_final.sort_values("score_final", ascending=False).reset_index(drop=True)
        pdf_final["rank_final"] = (pdf_final["score_final"].rank(method="first", ascending=False)).astype(int)

        features_ranked = pdf_final.sort_values("rank_final")["feature"].tolist()
        feature_sets = {f"top_{k}": features_ranked[:k] for k in TOPK_LIST}
        feature_sets["all"] = features_ranked

        log_pandas_csv(pdf_final, "summary/feature_ranking_final.csv")
        mlflow.log_dict({"features_ranked": features_ranked}, "summary/features_ranked.json")
        mlflow.log_dict(feature_sets, "summary/topk_sets.json")
        mlflow.log_dict({"method_info": method_info, "methods": ordered_methods}, "summary/methods_summary.json")

        FS_FEATURES_RANKED = features_ranked
        FS_FEATURE_SETS = feature_sets

print("✅ Feature Selection ok")
print("• run:", RUN_FS_EXEC)
print("• top_5:", FS_FEATURE_SETS.get("top_5", []))

# COMMAND ----------

# MAGIC %md
# MAGIC ## T_TREINO

# COMMAND ----------

import os, json, tempfile
import pandas as pd

import mlflow
import mlflow.spark

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# -------------------------
# helpers locais — TREINO
# -------------------------
def compute_logloss(df_pred, label_col="label", prob_col="probability", eps=1e-15):
    p1 = vector_to_array(F.col(prob_col)).getItem(1)
    p1c = F.least(F.greatest(p1, F.lit(eps)), F.lit(1.0 - eps))
    y = F.col(label_col).cast("double")
    ll = df_pred.select(
        (- (y * F.log(p1c) + (1.0 - y) * F.log(1.0 - p1c))).alias("ll")
    ).agg(F.avg("ll").alias("logloss")).collect()[0]["logloss"]
    return float(ll)

def log_pandas_csv(pdf: pd.DataFrame, artifact_path: str):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        pdf.to_csv(fp, index=False)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

def blank_to_null_all_strings(df: DataFrame) -> DataFrame:
    out = df
    for c, t in out.dtypes:
        if t == "string":
            out = out.withColumn(c, F.when(F.length(F.trim(F.col(c))) == 0, F.lit(None)).otherwise(F.col(c)))
    return out

def cast_numeric_to_double(df: DataFrame, cols: list) -> DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast("double"))
    return out

def build_threshold_metrics(df_pred: DataFrame, thresholds: list, label_col="label", prob_col="probability") -> pd.DataFrame:
    p1 = vector_to_array(F.col(prob_col)).getItem(1).cast("double")
    y = F.col(label_col).cast("int")

    rows = []
    for t in thresholds:
        yhat = (p1 >= F.lit(float(t))).cast("int")
        agg = (df_pred.select(y.alias("y"), yhat.alias("yhat"))
                     .agg(
                         F.sum(F.when((F.col("y")==1) & (F.col("yhat")==1), 1).otherwise(0)).alias("tp"),
                         F.sum(F.when((F.col("y")==0) & (F.col("yhat")==1), 1).otherwise(0)).alias("fp"),
                         F.sum(F.when((F.col("y")==0) & (F.col("yhat")==0), 1).otherwise(0)).alias("tn"),
                         F.sum(F.when((F.col("y")==1) & (F.col("yhat")==0), 1).otherwise(0)).alias("fn"),
                         F.count(F.lit(1)).alias("n"),
                     )
                     .collect()[0])
        tp, fp, tn, fn, n = int(agg["tp"]), int(agg["fp"]), int(agg["tn"]), int(agg["fn"]), int(agg["n"])
        prec = (tp / (tp+fp)) if (tp+fp) > 0 else None
        rec  = (tp / (tp+fn)) if (tp+fn) > 0 else None
        rows.append({"threshold": float(t), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n, "precision": prec, "recall": rec})
    return pd.DataFrame(rows)


# -------------------------
# T_TREINO (container limpo) + exec run versionada
# -------------------------
RUN_TR_GBT_TOP5 = run_name_vts("T_TREINO")

_treino_kw = {"run_id": TREINO_RUN_ID_OVERRIDE} if TREINO_RUN_ID_OVERRIDE else {"run_name": STEP_TREINO_NAME}
with mlflow.start_run(**_treino_kw, nested=True) as treino_container:
    TREINO_CONTAINER_RUN_ID = treino_container.info.run_id

    with mlflow.start_run(run_name=RUN_TR_GBT_TOP5, nested=True) as r:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("stage", "TREINO")
        mlflow.set_tag("run_role", "exec")
        mlflow.set_tag("mode", MODE_CODE)
        mlflow.set_tag("step", "TREINO")
        mlflow.set_tag("algo", "gbt")
        mlflow.set_tag("treino_versao", TREINO_VERSAO)
        mlflow.set_tag("versao_ref", VERSAO_REF)

        # contexto + lineage
        mlflow.log_param("ts_exec", TS_EXEC)
        mlflow.log_param("treino_versao", TREINO_VERSAO)
        mlflow.log_param("mode_code", MODE_CODE)
        mlflow.log_param("run_suffix", RUN_SUFFIX)

        mlflow.log_param("df_model_fqn", DF_MODEL_FQN)
        mlflow.log_param("seg_target", SEG_TARGET)
        mlflow.log_param("seed", int(SEED))
        mlflow.log_param("train_frac_internal", float(TRAIN_FRAC))
        mlflow.log_param("precision_target", float(PRECISION_TARGET))

        mlflow.log_param("pr_run_id", PR_RUN_ID)
        mlflow.log_param("mode_run_id", MODE_RUN_ID)
        mlflow.log_param("t_treino_container_run_id", TREINO_CONTAINER_RUN_ID)

        # -------------------------
        # Seleciona TOP5 features (precisa ter rodado FS)
        # -------------------------
        if "FS_FEATURE_SETS" not in globals():
            raise ValueError("❌ FS_FEATURE_SETS não encontrado. Execute a Feature Selection antes do treino.")
        selected_features = FS_FEATURE_SETS.get("top_5")
        if not selected_features or len(selected_features) != 5:
            raise ValueError(f"❌ top_5 inválido: {selected_features}")

        mlflow.log_dict({"selected_features_top5": selected_features}, "features/selected_features_used.json")
        mlflow.log_param("n_selected_features_base", int(len(selected_features)))

        # -------------------------
        # Load df_model + filtra SEG_TARGET
        # -------------------------
        df_raw = spark.table(DF_MODEL_FQN)
        df_seg = df_raw.filter(F.col(SEG_COL) == F.lit(SEG_TARGET)).cache()
        n_rows = df_seg.count()
        mlflow.log_metric("n_rows_seg", int(n_rows))

        df_lab = df_seg.withColumn("label_int", F.col(LABEL_COL).cast("int"))
        invalid = df_lab.filter(~F.col("label_int").isin([0, 1]) | F.col("label_int").isNull()).count()
        mlflow.log_metric("n_label_invalid_or_null", int(invalid))

        df_base = df_lab.drop(LABEL_COL).withColumnRenamed("label_int", "label")
        df_base = blank_to_null_all_strings(df_base)

        missing_feats = [c for c in selected_features if c not in df_base.columns]
        if missing_feats:
            raise ValueError(f"❌ Features do top5 não existem no df_model filtrado: {missing_feats}")

        dtypes = dict(df_base.select(*selected_features).dtypes)
        cat_feats = [c for c in selected_features if dtypes.get(c) == "string"]
        num_feats = [c for c in selected_features if dtypes.get(c) != "string"]

        mlflow.log_dict({"cat_features": cat_feats, "num_features": num_feats}, "features/feature_types.json")

        df_base = cast_numeric_to_double(df_base, num_feats)

        # -------------------------
        # Split interno train/val (estratificado por label)
        # -------------------------
        df_audit = df_base.select(ID_COL, "label", *selected_features).cache()
        fractions = {0: TRAIN_FRAC, 1: TRAIN_FRAC}
        df_train_audit = df_audit.sampleBy("label", fractions=fractions, seed=SEED).cache()
        train_ids = df_train_audit.select(ID_COL).cache()
        df_val_audit = df_audit.join(train_ids, on=ID_COL, how="left_anti").cache()

        n_tr = df_train_audit.count()
        n_va = df_val_audit.count()
        mlflow.log_metric("n_train_internal", int(n_tr))
        mlflow.log_metric("n_val_internal", int(n_va))

        df_train_ml = df_train_audit.select("label", *selected_features)
        df_val_ml   = df_val_audit.select("label", *selected_features)

        # -------------------------
        # Pipeline completo (preprocess + GBT) — vai para inferência
        # -------------------------
        idx_cols = [f"{c}__idx" for c in cat_feats]
        ohe_cols = [f"{c}__ohe" for c in cat_feats]
        imp_cols = [f"{c}__imp" for c in num_feats]

        stages = []
        for c, out_c in zip(cat_feats, idx_cols):
            stages.append(StringIndexer(inputCol=c, outputCol=out_c, handleInvalid="keep"))
        if idx_cols:
            stages.append(OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, dropLast=False))
        if num_feats:
            stages.append(Imputer(inputCols=num_feats, outputCols=imp_cols, strategy="mean"))

        assembler_inputs = ohe_cols + imp_cols
        stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features_vec"))

        gbt = GBTClassifier(
            featuresCol="features_vec",
            labelCol="label",
            maxIter=80,
            maxDepth=5,
            stepSize=0.1,
            seed=SEED,
        )

        mlflow.log_params({"gbt_maxIter": 80, "gbt_maxDepth": 5, "gbt_stepSize": 0.1, "gbt_seed": int(SEED)})

        pipeline = Pipeline(stages=stages + [gbt])
        pipeline_model = pipeline.fit(df_train_ml)

        pred_val = pipeline_model.transform(df_val_ml)

        evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        evaluator_pr  = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

        auc = float(evaluator_auc.evaluate(pred_val))
        prauc = float(evaluator_pr.evaluate(pred_val))
        logloss = float(compute_logloss(pred_val))

        mlflow.log_metrics({"auc_val": auc, "prauc_val": prauc, "logloss_val": logloss})

        # -------------------------
        # Threshold grid
        # -------------------------
        pdf_thr = build_threshold_metrics(pred_val.select("label", "probability"), THRESHOLDS, label_col="label", prob_col="probability")

        pdf_ok = pdf_thr.dropna(subset=["precision", "recall"])
        pdf_ok = pdf_ok[pdf_ok["precision"] >= PRECISION_TARGET]
        rec_threshold = None
        if not pdf_ok.empty:
            best = pdf_ok.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
            rec_threshold = float(best["threshold"])
            mlflow.log_param("recommended_threshold_precision_ge_target", rec_threshold)

        log_pandas_csv(pdf_thr, "threshold/threshold_metrics.csv")
        mlflow.log_dict(
            {"threshold_metrics": pdf_thr.to_dict(orient="records"), "recommended_threshold": rec_threshold, "precision_target": PRECISION_TARGET},
            "threshold/threshold_metrics.json",
        )

        # -------------------------
        # Log do modelo para inferência
        # -------------------------
        MODEL_ARTIFACT_PATH = "model"
        mlflow.log_param("model_artifact_path", MODEL_ARTIFACT_PATH)

        mlflow.spark.log_model(pipeline_model, artifact_path=MODEL_ARTIFACT_PATH)

        MODEL_URI_FOR_INFERENCE = f"runs:/{r.info.run_id}/{MODEL_ARTIFACT_PATH}"
        mlflow.log_param("model_uri_for_inference", MODEL_URI_FOR_INFERENCE)

        summary = {
            "model": "GBT",
            "seg_target": SEG_TARGET,
            "selected_features_top5": selected_features,
            "cat_features": cat_feats,
            "num_features": num_feats,
            "auc_val": auc,
            "prauc_val": prauc,
            "logloss_val": logloss,
            "recommended_threshold_precision_ge_target": rec_threshold,
            "model_uri_for_inference": MODEL_URI_FOR_INFERENCE,
            "note": "Modelo logado como PipelineModel (preprocess + gbt) em artifact_path='model'.",
        }
        mlflow.log_dict(summary, "training_summary.json")

print("✅ T_TREINO concluído (modelo logado para inferência)")
print("• run:", RUN_TR_GBT_TOP5)
print("• use este MODEL_URI:", MODEL_URI_FOR_INFERENCE)