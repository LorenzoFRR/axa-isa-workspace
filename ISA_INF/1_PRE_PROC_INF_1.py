# Databricks notebook source
# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import json
from typing import Callable, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

mlflow.autolog(disable=True)   # evita runs automáticas do Databricks autolog que quebram o aninhamento

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from functools import reduce

# COMMAND ----------

# MAGIC %md
# MAGIC # Configs

# COMMAND ----------

# =========================
# AJUSTES RECORRENTES
# =========================
PR_RUN_ID_OVERRIDE = "bb7f8f463fd94a85a2040c58ec2baf60"

# =========================
# MLflow
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/ISA_INF (Lorenzo)/ISA_INF"
PARENT_RUN_NAME = "I_PR_PRE_PROC"

# =========================
# Versão/timestamp da etapa
# =========================
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")

# =========================
# Inputs (bronze) — tabelas fixas do pipeline de inferência
# =========================
BRONZE_FACT_FQN             = "bronze.cotacao_generico_inferencia"
BRONZE_CORRETOR_RESUMO_FQN  = "bronze.corretor_resumo_inferencia"
BRONZE_CORRETOR_DETALHE_FQN = "bronze.corretor_detalhe_inferencia"

# =========================
# Outputs (silver) — tabelas fixas, com FL_NOVO para controle incremental
# =========================
SILVER_SCHEMA = "silver"

SILVER_FACT_FQN             = f"{SILVER_SCHEMA}.cotacao_generico_clean_inf"
SILVER_CORRETOR_RESUMO_FQN  = f"{SILVER_SCHEMA}.corretor_resumo_clean_inf"
SILVER_CORRETOR_DETALHE_FQN = f"{SILVER_SCHEMA}.corretor_detalhe_clean_inf"

# =========================
# Flags para desenvolvimento dinâmico de regras
# =========================
ENABLE_RULES = True

# =========================
# Colunas técnicas
# =========================
TECH_COLS = ["TS_ARQ", "TS_ATUALIZACAO", "SOURCE_FILE"]

print("CONFIG 1_PRE_PROC_INF carregada")
print("• bronze fact:", BRONZE_FACT_FQN, "(FL_NOVO == True)")
print("• silver fact:", SILVER_FACT_FQN, "(tabela fixa)\n")

print("• bronze resumo:", BRONZE_CORRETOR_RESUMO_FQN, "(FL_NOVO == True)")
print("• silver resumo:", SILVER_CORRETOR_RESUMO_FQN, "(tabela fixa)\n")

print("• bronze detalhe:", BRONZE_CORRETOR_DETALHE_FQN, "(FL_NOVO == True)")
print("• silver detalhe:", SILVER_CORRETOR_DETALHE_FQN, "(tabela fixa)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Define helpers

# COMMAND ----------

# =========================
# Helpers: metastore / schemas
# =========================
def ensure_schema(schema_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

def table_exists(fqn: str) -> bool:
    return spark.catalog.tableExists(fqn)

def assert_table_exists(fqn: str) -> None:
    if not table_exists(fqn):
        raise ValueError(f"Tabela não existe: {fqn}")


# =========================
# Helpers: MLflow experiment
# =========================
def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


# =========================
# Helpers: transformações genéricas
# =========================
def safe_drop_cols(df: DataFrame, cols: List[str]) -> DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(*existing) if existing else df

def null_if_blank(col: F.Column) -> F.Column:
    return F.when(F.trim(col) == F.lit(""), F.lit(None)).otherwise(col)

def normalize_ptbr_number_str(col: F.Column) -> F.Column:
    c = F.regexp_replace(col.cast("string"), r'["\s]', "")
    c = F.regexp_replace(c, r"\.", "")
    c = F.regexp_replace(c, r",", ".")
    return c

def cast_decimal(df: DataFrame, col_name: str, precision: int, scale: int) -> DataFrame:
    if col_name not in df.columns:
        return df
    dec_type = T.DecimalType(precision=precision, scale=scale)
    return df.withColumn(col_name, normalize_ptbr_number_str(F.col(col_name)).cast(dec_type))

def drop_all_null_or_blank_columns(df: DataFrame) -> DataFrame:
    exprs = []
    for c, t in df.dtypes:
        if t == "string":
            exprs.append(F.max(F.when(F.trim(F.col(c)) != "", F.lit(1)).otherwise(F.lit(0))).alias(c))
        else:
            exprs.append(F.max(F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(c))

    flags = df.agg(*exprs).collect()[0].asDict()
    keep = [c for c in df.columns if flags.get(c, 1) == 1]
    return df.select(*keep)

def auto_cast_numeric_cols(df: DataFrame) -> DataFrame:
    string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]
    if not string_cols:
        return df

    agg_exprs = []
    for c in string_cols:
        non_null = F.col(c).isNotNull() & (F.trim(F.col(c)) != "")
        agg_exprs += [
            F.count(F.when(non_null & ~F.col(c).rlike(r"^-?\d+$"),                         True)).alias(f"{c}__long_fail"),
            F.count(F.when(non_null & F.expr(f"try_cast(`{c}` as double)").isNull(), True)).alias(f"{c}__dbl_fail"),
        ]

    stats = df.agg(*agg_exprs).collect()[0].asDict()

    for c in string_cols:
        if stats[f"{c}__long_fail"] == 0:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` as long)"))
        elif stats[f"{c}__dbl_fail"] == 0:
            df = df.withColumn(c, F.expr(f"try_cast(`{c}` as double)"))

    return df


# =========================
# Modelo de regra + executor dinâmico
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

def _is_rule_enabled(table_key: str, rule_id: str, base_enabled: bool, toggles: Dict) -> bool:
    if not base_enabled:
        return False
    if not toggles:
        return True
    t = toggles.get(table_key, {})
    if rule_id in t:
        return bool(t[rule_id])
    return True

def apply_rules_for_table(
    table_key: str,
    df: DataFrame,
    rules: List[Dict],
    enable_rules: bool = True,
    rule_toggles: Optional[Dict] = None,
) -> Tuple[DataFrame, List[Dict]]:
    exec_log: List[Dict] = []
    rule_toggles = rule_toggles or {}

    df_out = df
    for r in rules:
        rid = r["rule_id"]
        desc = r["description"]
        req = r.get("requires_columns", []) or []

        enabled = enable_rules and _is_rule_enabled(table_key, rid, r.get("enabled", True), rule_toggles)
        if not enabled:
            exec_log.append({"rule_id": rid, "status": "SKIPPED_DISABLED", "description": desc})
            continue

        missing = [c for c in req if c not in df_out.columns]
        if missing:
            exec_log.append({
                "rule_id": rid,
                "status": "SKIPPED_MISSING_COLS",
                "description": desc,
                "reason": f"missing_columns={missing}",
            })
            continue

        try:
            df_out = r["fn"](df_out)
            exec_log.append({"rule_id": rid, "status": "APPLIED", "description": desc})
        except Exception as e:
            raise RuntimeError(f"Falha ao aplicar regra {table_key}.{rid}: {desc}. Erro: {e}") from e

    return df_out, exec_log


# =========================
# Helpers: logging (catálogo, lineage)
# =========================
def build_rules_catalog_for_logging(rules_by_table: Dict[str, List[Dict]]) -> Dict:
    out = {}
    for table_key, rules in rules_by_table.items():
        out[table_key] = [
            {
                "rule_id": r["rule_id"],
                "description": r["description"],
                "enabled": bool(r.get("enabled", True)),
                "requires_columns": r.get("requires_columns", []) or [],
            }
            for r in rules
        ]
    return out

def build_tables_lineage_dict() -> Dict:
    return {
        "cotacao_generico": {"bronze": BRONZE_FACT_FQN, "silver": SILVER_FACT_FQN},
        "corretor_resumo":  {"bronze": BRONZE_CORRETOR_RESUMO_FQN, "silver": SILVER_CORRETOR_RESUMO_FQN},
        "corretor_detalhe": {"bronze": BRONZE_CORRETOR_DETALHE_FQN, "silver": SILVER_CORRETOR_DETALHE_FQN},
        "ts_exec": TS_EXEC,
    }


print("CEL 1 carregada: helpers + executor de regras prontos")


# =========================================================
# Helpers específicos
# =========================================================
def clean_numeric_str_us(colname: str) -> F.Column:
    s = F.trim(F.col(colname).cast("string"))
    s = F.when(s.isNull() | (s == F.lit("")), F.lit(None)).otherwise(s)
    s = F.regexp_replace(s, ",", "")
    s = F.regexp_replace(s, r"^\.", "0.")
    s = F.regexp_replace(s, r"^-\.", "-0.")
    return s

def cast_decimal_us(df: DataFrame, cols: List[str], scale: int) -> DataFrame:
    dtype = T.DecimalType(17, scale)
    for c in cols:
        if c in df.columns:
            s = clean_numeric_str_us(c)
            df = df.withColumn(c, F.round(s.cast("double"), scale).cast(dtype))
    return df

def parse_to_date_any(cname: str) -> F.Column:
    """Tenta parsear uma coluna para DateType testando múltiplos formatos.
    Usa TRY_TO_TIMESTAMP (nunca lança exceção em ANSI mode) em vez de to_timestamp.
    Formatos ISO8601 com 'T' literal foram removidos: Photon não suporta literais
    entre aspas simples em format strings, e esses formatos não ocorrem nos dados."""
    s = f"CAST(`{cname}` AS STRING)"
    return F.coalesce(
        F.expr(f"TRY_TO_TIMESTAMP({s}, 'yyyy-MM-dd HH:mm:ss.SSS')").cast("date"),
        F.expr(f"TRY_TO_TIMESTAMP({s}, 'yyyy-MM-dd HH:mm:ss')").cast("date"),
        F.expr(f"TRY_TO_TIMESTAMP({s}, 'yyyy-MM-dd')").cast("date"),
        F.expr(f"TRY_TO_TIMESTAMP({s}, 'dd/MM/yyyy HH:mm:ss')").cast("date"),
        F.expr(f"TRY_TO_TIMESTAMP({s}, 'dd/MM/yyyy')").cast("date"),
        F.expr(f"TRY_CAST({s} AS DATE)"),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Regras de negócio / Pré-Processamento

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regras COTACAO_GENERICO

# COMMAND ----------

def GEN_R01_data_cotacao(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "DATA_COTACAO",
        F.make_date(
            F.col("ANO_COTACAO").cast("int"),
            F.col("MES_COTACAO").cast("int"),
            F.col("DIA_COTACAO").cast("int"),
        )
    )

def GEN_R02_canal(df: DataFrame) -> DataFrame:
    fl_str = F.lower(F.trim(F.col("FL_ANALISE_SUBSCRICAO").cast("string")))
    fl_manual = fl_str.isin("sim", "true", "1", "yes")
    cot_is_c = F.upper(F.trim(F.col("CD_NUMERO_COTACAO_AXA").cast("string"))).startswith("C")

    return df.withColumn(
        "CANAL",
        F.when(fl_manual | cot_is_c, F.lit("MANUAL")).otherwise(F.lit("DIGITAL"))
    )

def GEN_R03_dt_inicio_vigencia_null(df: DataFrame) -> DataFrame:
    dt = parse_to_date_any("DT_INICIO_VIGENCIA")
    return df.withColumn(
        "DT_INICIO_VIGENCIA",
        F.when(
            (F.year(dt) > 2027) | (dt == F.lit("1900-01-01").cast("date")),
            F.lit(None).cast("date"),
        ).otherwise(dt)
    )

def GEN_R04_ajuste_numerico_vl(df: DataFrame) -> DataFrame:
    cols = ["VL_ENDOSSO_PREMIO_TOTAL", "VL_PREMIO_ALVO", "VL_PREMIO_LIQUIDO", "VL_PRE_TOTAL"]
    return cast_decimal_us(df, cols, scale=2)

def GEN_R05_ds_tipo_cotacao(df: DataFrame) -> DataFrame:
    c = F.col("DS_TIPO_COTACAO")
    return df.withColumn(
        "DS_TIPO_COTACAO",
        F.when(F.lower(F.trim(c)) == F.lit("renovação congênere"), F.lit("Seguro Novo"))
         .when(c.isNull() | (F.trim(c) == ""), F.lit("Seguro Novo"))
         .otherwise(c)
    )

def GEN_R06_filtra_recusados_e_regra_C(df: DataFrame) -> DataFrame:
    cot_norm    = F.upper(F.trim(F.col("CD_NUMERO_COTACAO_AXA").cast("string")))
    status_norm = F.upper(F.trim(F.coalesce(F.col("DS_STATUS").cast("string"), F.lit(""))))
    salvo_norm  = F.upper(F.trim(F.coalesce(F.col("FL_SALVO").cast("string"), F.lit(""))))

    recusados = ["RECUSADA PELO SUBSCRITOR", "RECUSADA PELO CORRETOR"]
    df_base = df.filter(~status_norm.isin(recusados))

    df_c = df_base.filter(cot_norm.startswith("C"))
    df_non_c = df_base.filter(~cot_norm.startswith("C")).filter(salvo_norm == F.lit("SIM"))

    return df_c.unionByName(df_non_c)

def GEN_R07_regra_temporal(df: DataFrame) -> DataFrame:
    df1 = (
        df
        .withColumn("TS_ARQ_ts", F.col("TS_ARQ").cast("timestamp"))
        .withColumn("STATUS_NORM", F.upper(F.trim(F.col("DS_GRUPO_STATUS").cast("string"))))
    )

    # mantém último snapshot por cotação dentro do batch FL_NOVO
    w = Window.partitionBy("CD_NUMERO_COTACAO_AXA").orderBy(F.col("TS_ARQ_ts").desc_nulls_last())
    df_latest = df1.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

    ref_ts = df_latest.agg(F.max("TS_ARQ_ts").alias("ref_ts")).first()["ref_ts"]

    df2 = (
        df_latest
        .withColumn("REF_TS", F.lit(ref_ts).cast("timestamp"))
        .withColumn("DIAS_ULTIMA_ATUALIZACAO", F.datediff(F.to_date("REF_TS"), F.to_date("TS_ARQ_ts")))
    )

    status_alvo = ["EM ANALISE", "EM ANÁLISE", "EM NEGOCIACAO", "EM NEGOCIAÇÃO"]
    cond_perdida = F.col("STATUS_NORM").isin(status_alvo) & (F.col("DIAS_ULTIMA_ATUALIZACAO") > F.lit(90))

    df_out = (
        df2
        .withColumn("DS_GRUPO_STATUS_OLD", F.col("DS_GRUPO_STATUS"))
        .withColumn("DS_GRUPO_STATUS", F.when(cond_perdida, F.lit("Perdida")).otherwise(F.col("DS_GRUPO_STATUS")))
        .drop("STATUS_NORM")
    )

    # histórico para detectar "primeira vez" em EMITIDA/PERDIDA
    w_hist = Window.partitionBy("CD_NUMERO_COTACAO_AXA").orderBy(F.col("TS_ARQ_ts").asc())
    df_hist2 = df1.withColumn("prev_status", F.lag("STATUS_NORM").over(w_hist))

    df_fechamento_emitida = (
        df_hist2
        .filter((F.col("STATUS_NORM") == "EMITIDA") & (F.col("prev_status").isNull() | (F.col("prev_status") != "EMITIDA")))
        .groupBy("CD_NUMERO_COTACAO_AXA")
        .agg(F.min("TS_ARQ_ts").alias("TS_FECHAMENTO_EMITIDA"))
    )

    df_fechamento_perdida_nat = (
        df_hist2
        .filter((F.col("STATUS_NORM") == "PERDIDA") & (F.col("prev_status").isNull() | (F.col("prev_status") != "PERDIDA")))
        .groupBy("CD_NUMERO_COTACAO_AXA")
        .agg(F.min("TS_ARQ_ts").alias("TS_FECHAMENTO_PERDIDA_NAT"))
    )

    df_out = df_out.withColumn(
        "FL_PERDIDA_90D",
        (F.upper(F.trim(F.col("DS_GRUPO_STATUS").cast("string"))) == F.lit("PERDIDA")) &
        (F.upper(F.trim(F.col("DS_GRUPO_STATUS_OLD").cast("string"))).isin([s.upper() for s in status_alvo])) &
        (F.col("DIAS_ULTIMA_ATUALIZACAO") > F.lit(90))
    )

    df_out = (
        df_out
        .join(df_fechamento_emitida, on="CD_NUMERO_COTACAO_AXA", how="left")
        .join(df_fechamento_perdida_nat, on="CD_NUMERO_COTACAO_AXA", how="left")
    )

    status_final = F.upper(F.trim(F.col("DS_GRUPO_STATUS").cast("string")))

    df_out = df_out.withColumn(
        "REF_DATE",
        F.when(status_final == F.lit("EMITIDA"), F.to_date("TS_FECHAMENTO_EMITIDA"))
         .when((status_final == F.lit("PERDIDA")) & (F.col("FL_PERDIDA_90D") == F.lit(True)), F.to_date("TS_ARQ_ts"))
         .when(status_final == F.lit("PERDIDA"), F.to_date("TS_FECHAMENTO_PERDIDA_NAT"))
         .otherwise(F.to_date("TS_ARQ_ts"))
    )

    map_colunas = {
        "DT_VALIDADE": "DIAS_VALIDADE",
        "DT_ANALISE_SUBSCRICAO": "DIAS_ANALISE_SUBSCRICAO",
        "DT_FIM_ANALISE_SUBSCRICAO": "DIAS_FIM_ANALISE_SUBSCRICAO",
    }

    df_out2 = df_out
    for old, new in map_colunas.items():
        if old in df_out2.columns:
            d = parse_to_date_any(old)
            df_out2 = df_out2.withColumn(new, F.datediff(d, F.col("REF_DATE")).cast("int")).drop(old)

    if "DATA_COTACAO" in df_out2.columns:
        df_out2 = df_out2.withColumn("DATA_COTACAO", parse_to_date_any("DATA_COTACAO"))
        df_out2 = df_out2.withColumn("DIAS_COTACAO", F.datediff(F.col("DATA_COTACAO"), F.col("REF_DATE")).cast("int"))

    if "DT_INICIO_VIGENCIA" in df_out2.columns and "DATA_COTACAO" in df_out2.columns:
        df_out2 = df_out2.withColumn("DT_INICIO_VIGENCIA", parse_to_date_any("DT_INICIO_VIGENCIA"))
        df_out2 = df_out2.withColumn(
            "DIAS_INICIO_VIGENCIA",
            F.datediff(F.col("DT_INICIO_VIGENCIA"), F.col("DATA_COTACAO")).cast("int")
        )

    return df_out2

def GEN_R08_drop_colunas(df: DataFrame) -> DataFrame:
    cols_drop_gen = [
        "TS_ATUALIZACAO",
        "SOURCE_FILE",
        "DS_MOTIVO_ENDOSSO",
        "DS_REPIQUE_ACAO",
        "DS_REPIQUE_MOTIVO",
        "DS_REPIQUE_ATENDIMENTO",
        "FL_PROPOSTA",
        "FL_REPIQUE",
        "FL_ENDOSSO",
        "DS_FAROL",
        "ANO_COTACAO",
        "MES_COTACAO",
        "DIA_COTACAO",
        "TS_ARQ_ts",
        "REF_TS",
        "DS_GRUPO_STATUS_OLD",
        "DS_TIPO_SEGURO",
        "CD_NUM_PROPOSTA",
        "CD_NUMERO_APOLICE_AXA",
        "CD_NUMERO_ENDOSSO_AXA",
        "FL_ENDOSSO_RESTITUICAO",
        "CD_FILIAL_AXA",
        "DS_NOME_VERSAO_CALCULO",
        "DS_STATUS",
        "DS_CORRETOR_SEGMENTO",
        "DS_SUBSCRITOR",
        "FL_ANALISE_SUBSCRICAO",
        "FL_SALVO",
        "FL_PERDIDA_90D",
        "REF_DATE",
        "TS_FECHAMENTO_EMITIDA",
        "TS_FECHAMENTO_PERDIDA_NAT",
    ]
    return safe_drop_cols(df, cols_drop_gen)

def GEN_R09_normaliza_status(df: DataFrame) -> DataFrame:
    s = F.upper(F.trim(F.col("DS_GRUPO_STATUS").cast("string")))
    return df.withColumn(
        "DS_GRUPO_STATUS",
        F.when(s == "EMITIDA", F.lit("Emitida"))
         .when(s == "PERDIDA", F.lit("Perdida"))
         .otherwise(F.col("DS_GRUPO_STATUS"))
    )

def GEN_R12_cria_mes(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("DATA_COTACAO_dt", parse_to_date_any("DATA_COTACAO"))
          .filter(F.col("DATA_COTACAO_dt").isNotNull())
          .withColumn("MES", F.date_format("DATA_COTACAO_dt", "yyyy-MM"))
          .drop("DATA_COTACAO_dt")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regras CORRETOR_RESUMO

# COMMAND ----------

def RES_R01_dedupe_cd_corretor(df: DataFrame) -> DataFrame:
    key = "CD_CORRETOR"
    cols_to_score = [c for c in df.columns if c != key]

    filled_flags = [
        F.when(F.col(c).isNull(), F.lit(0))
         .when(F.trim(F.col(c).cast("string")) == F.lit(""), F.lit(0))
         .otherwise(F.lit(1))
        for c in cols_to_score
    ]

    df_scored = df.withColumn("filled_score", reduce(lambda a, b: a + b, filled_flags, F.lit(0)))

    order_cols = [F.col("filled_score").desc()]
    if "TS_ARQ" in df.columns:
        order_cols.append(F.col("TS_ARQ").desc_nulls_last())

    w = Window.partitionBy(key).orderBy(*order_cols)
    return (
        df_scored
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn", "filled_score")
    )

def RES_R02_vl_gwp_corretor_decimal(df: DataFrame) -> DataFrame:
    return cast_decimal_us(df, ["VL_GWP_CORRETOR"], scale=2)

def RES_R03_ds_canal_comercial(df: DataFrame) -> DataFrame:
    c = F.col("DS_CANAL_COMERCIAL").cast("string")
    return df.withColumn(
        "DS_CANAL_COMERCIAL",
        F.when(c.isNull() | (F.trim(c) == ""), F.lit("MANUAL"))
         .when(F.upper(F.trim(c)) == "DIGITAL", F.lit("DIGITAL"))
         .otherwise(F.lit("MANUAL"))
    )

def RES_R04_drop_cols_tecnicas(df: DataFrame) -> DataFrame:
    cols_tech = ["TS_ARQ", "TS_ATUALIZACAO", "SOURCE_FILE", "DS_GRUPO_CORRETOR", "DT_APROVACAO_CADASTRO"]
    return safe_drop_cols(df, cols_tech)

def RES_R05_drop_cols_100pct_vazias(df: DataFrame) -> DataFrame:
    return drop_all_null_or_blank_columns(df)

def RES_R06_drop_colunas_extras(df: DataFrame) -> DataFrame:
    cols = [
        "DS_CORRETOR",
        "DS_SEGMENTACAO",
        "DS_SUCURSAL_AUTORIZA_EMISSAO",
        "DS_SEGMENTACAO_GRUPO",
        "DS_CANAL_COMERCIAL",
    ]
    return safe_drop_cols(df, cols)

def RES_R07_drop_colunas_flag(df: DataFrame) -> DataFrame:
    cols = [
        "FL_PERMITE_ANTECIPAR",
        "FL_ACORDO_GENERALISTA",
        "FL_ACORDO_ESPECIFICO",
    ]
    return safe_drop_cols(df, cols)

def RES_R08_uppercase_cols(df: DataFrame) -> DataFrame:
    return df.toDF(*[c.upper() for c in df.columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regras CORRETOR_DETALHE

# COMMAND ----------

def DET_R01_ds_tipo_solicitacao_default(df: DataFrame) -> DataFrame:
    c = F.col("DS_TIPO_SOLICITACAO").cast("string")
    return df.withColumn(
        "DS_TIPO_SOLICITACAO",
        F.when(c.isNull() | (F.trim(c) == ""), F.lit("Seguro Novo")).otherwise(F.col("DS_TIPO_SOLICITACAO"))
    )

def DET_R02_drop_cols_tecnicas(df: DataFrame) -> DataFrame:
    return safe_drop_cols(df, ["TS_ARQ", "TS_ATUALIZACAO", "SOURCE_FILE"])

def DET_R03_hr_decimals(df: DataFrame) -> DataFrame:
    cols = ["HR_2024", "HR_2025", "HR_M2", "HR_M3"]
    return cast_decimal_us(df, cols, scale=6)

# COMMAND ----------

# MAGIC %md
# MAGIC # Catálogo Dinâmico

# COMMAND ----------

# =========================================================
# Catálogo dinâmico (RULES_BY_TABLE)
#
# Diferenças vs. 1_PRE_PROC.py (pipeline de treino):
#   GEN_R10 (filtra status finais) → disabled: inferência processa cotações em aberto
#   GEN_R11 (cria label)           → disabled: não há desfecho conhecido em inferência
# =========================================================
RULES_BY_TABLE = {
    "cotacao_generico": [
        rule_def("GEN_R01", "Criar DATA_COTACAO a partir de ANO/MES/DIA_COTACAO", GEN_R01_data_cotacao,
                 enabled=True,
                 requires_columns=["ANO_COTACAO", "MES_COTACAO", "DIA_COTACAO"]),
        rule_def("GEN_R02", "Definir CANAL (MANUAL se FL_ANALISE_SUBSCRICAO indicar manual OU cotação começa com 'C'; senão DIGITAL)",
                 GEN_R02_canal, enabled=True,
                 requires_columns=["FL_ANALISE_SUBSCRICAO", "CD_NUMERO_COTACAO_AXA"]),
        rule_def("GEN_R03", "Normalizar DT_INICIO_VIGENCIA inválida (ano>2027 ou 1900-01-01) para NULL",
                 GEN_R03_dt_inicio_vigencia_null, enabled=True,
                 requires_columns=["DT_INICIO_VIGENCIA"]),
        rule_def("GEN_R04", "Ajustar colunas VL_* para decimal(17,2) (limpeza + cast)",
                 GEN_R04_ajuste_numerico_vl, enabled=True),
        rule_def("GEN_R05", "Normalizar DS_TIPO_COTACAO: 'renovação congênere' e nulos/vazios -> 'Seguro Novo'",
                 GEN_R05_ds_tipo_cotacao, enabled=True,
                 requires_columns=["DS_TIPO_COTACAO"]),
        rule_def("GEN_R06", "Remover recusadas (DS_STATUS) e aplicar regra: manter todas 'C*' + não-'C*' somente se FL_SALVO='SIM'",
                 GEN_R06_filtra_recusados_e_regra_C, enabled=True,
                 requires_columns=["CD_NUMERO_COTACAO_AXA", "DS_STATUS", "FL_SALVO"]),
        rule_def("GEN_R07", "Aplicar regra temporal: último snapshot por cotação + Perdida 90d + datas de fechamento + DIAS_*",
                 GEN_R07_regra_temporal, enabled=True,
                 requires_columns=["CD_NUMERO_COTACAO_AXA", "TS_ARQ", "DS_GRUPO_STATUS"]),
        rule_def("GEN_R08", "Dropar colunas fora do escopo",
                 GEN_R08_drop_colunas, enabled=True),
        rule_def("GEN_R09", "Normalizar casing de DS_GRUPO_STATUS para valores canônicos (Emitida, Perdida)",
                 GEN_R09_normaliza_status, enabled=True,
                 requires_columns=["DS_GRUPO_STATUS"]),
        # GEN_R10 desabilitada: inferência inclui cotações em aberto (sem desfecho definido)
        rule_def("GEN_R10", "Filtrar apenas cotações com desfecho conhecido (Emitida ou Perdida) — DESABILITADA em inferência",
                 lambda df: df, enabled=False,
                 requires_columns=["DS_GRUPO_STATUS"]),
        # GEN_R11 desabilitada: não há label em inferência
        rule_def("GEN_R11", "Criar coluna label (Emitida=1, Perdida=0) — DESABILITADA em inferência",
                 lambda df: df, enabled=False,
                 requires_columns=["DS_GRUPO_STATUS"]),
        rule_def("GEN_R12", "Criar coluna MES=yyyy-MM a partir de DATA_COTACAO; remove linhas com DATA_COTACAO nula",
                 GEN_R12_cria_mes, enabled=True,
                 requires_columns=["DATA_COTACAO"]),
    ],
    "corretor_resumo": [
        rule_def("RES_R01", "Remover duplicados por CD_CORRETOR escolhendo linha mais preenchida (filled_score) e desempate por TS_ARQ",
                 RES_R01_dedupe_cd_corretor, enabled=True,
                 requires_columns=["CD_CORRETOR"]),
        rule_def("RES_R02", "Converter VL_GWP_CORRETOR para decimal(17,2)", RES_R02_vl_gwp_corretor_decimal,
                 enabled=True),
        rule_def("RES_R03", "Normalizar DS_CANAL_COMERCIAL: NULL/vazio->MANUAL; DIGITAL permanece; demais->MANUAL",
                 RES_R03_ds_canal_comercial, enabled=True,
                 requires_columns=["DS_CANAL_COMERCIAL"]),
        rule_def("RES_R04", "Dropar colunas técnicas e colunas específicas",
                 RES_R04_drop_cols_tecnicas, enabled=True),
        rule_def("RES_R05", "Dropar colunas 100% vazias (NULL ou '')", RES_R05_drop_cols_100pct_vazias,
                 enabled=True),
        rule_def("RES_R06", "Dropar colunas extras sem utilidade downstream",
                 RES_R06_drop_colunas_extras, enabled=True),
        rule_def("RES_R07", "Dropar colunas de flag sem utilidade downstream",
                 RES_R07_drop_colunas_flag, enabled=True),
        rule_def("RES_R08", "Garantir nomes de colunas em UPPERCASE",
                 RES_R08_uppercase_cols, enabled=True),
    ],
    "corretor_detalhe": [
        rule_def("DET_R01", "Normalizar DS_TIPO_SOLICITACAO: NULL/vazio -> 'Seguro Novo'",
                 DET_R01_ds_tipo_solicitacao_default, enabled=True,
                 requires_columns=["DS_TIPO_SOLICITACAO"]),
        rule_def("DET_R02", "Dropar colunas técnicas (TS_ARQ, TS_ATUALIZACAO, SOURCE_FILE)", DET_R02_drop_cols_tecnicas,
                 enabled=True),
        rule_def("DET_R03", "Converter HR_* para decimal(17,6)", DET_R03_hr_decimals,
                 enabled=True),
    ],
}

print("CEL 2 carregada: RULES_BY_TABLE definido")
print("• regras cotacao_generico:", len(RULES_BY_TABLE["cotacao_generico"]))
print("• regras corretor_resumo:", len(RULES_BY_TABLE["corretor_resumo"]))
print("• regras corretor_detalhe:", len(RULES_BY_TABLE["corretor_detalhe"]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Executa MLflow

# COMMAND ----------

# =========================
# Validações e bootstrap de schemas
# =========================
ensure_schema(SILVER_SCHEMA)

assert_table_exists(BRONZE_FACT_FQN)
assert_table_exists(BRONZE_CORRETOR_RESUMO_FQN)
assert_table_exists(BRONZE_CORRETOR_DETALHE_FQN)

# Verifica se há dados novos para processar
n_novos_bronze = spark.table(BRONZE_FACT_FQN).filter(F.col("FL_NOVO") == True).count()
if n_novos_bronze == 0:
    print(f"⚠ Nenhuma linha FL_NOVO=True em {BRONZE_FACT_FQN}. Nada a fazer.")
    dbutils.notebook.exit("SKIP: sem dados novos")
print(f"• linhas FL_NOVO=True em bronze: {n_novos_bronze}")

# =========================
# MLflow bootstrap
# =========================
_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

child_run_name = f"I_PRE_PROC_{TS_EXEC}"

rules_catalog_for_logging = build_rules_catalog_for_logging(RULES_BY_TABLE)
tables_lineage = build_tables_lineage_dict()

rules_execution_log: Dict[str, List[Dict]] = {}
tables_written: Dict[str, str] = {}

n_rules_total   = sum(len(v) for v in RULES_BY_TABLE.values())
n_rules_applied = 0
n_rules_skipped = 0


# =========================
# Execução
# =========================
pr_ctx = (
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    if PR_RUN_ID_OVERRIDE
    else mlflow.start_run(run_name=PARENT_RUN_NAME)
)

with pr_ctx as pr:
    if not PR_RUN_ID_OVERRIDE:
        mlflow.set_tag("pipeline_tipo", "I")
        mlflow.set_tag("etapa", "PRE_PROC")
        mlflow.set_tag("run_role", "parent")

    with mlflow.start_run(run_name=child_run_name, nested=True) as cr:
        mlflow.set_tag("pipeline_tipo", "I")
        mlflow.set_tag("etapa", "PRE_PROC")
        mlflow.set_tag("run_role", "child")

        mlflow.log_param("bronze_fact_fqn",             BRONZE_FACT_FQN)
        mlflow.log_param("bronze_corretor_resumo_fqn",  BRONZE_CORRETOR_RESUMO_FQN)
        mlflow.log_param("bronze_corretor_detalhe_fqn", BRONZE_CORRETOR_DETALHE_FQN)
        mlflow.log_param("silver_fact_fqn",             SILVER_FACT_FQN)
        mlflow.log_param("silver_corretor_resumo_fqn",  SILVER_CORRETOR_RESUMO_FQN)
        mlflow.log_param("silver_corretor_detalhe_fqn", SILVER_CORRETOR_DETALHE_FQN)
        mlflow.log_param("ts_exec",                     TS_EXEC)
        mlflow.log_param("enable_rules",                str(ENABLE_RULES))

        mlflow.log_dict(rules_catalog_for_logging, "rules_catalog.json")
        mlflow.log_dict(tables_lineage, "tables_lineage.json")

        # -------------------------
        # 1) cotacao_generico — apenas FL_NOVO=True
        # -------------------------
        df_fact_in = (
            spark.table(BRONZE_FACT_FQN)
                 .filter(F.col("FL_NOVO") == True)
        )
        df_fact_in = auto_cast_numeric_cols(df_fact_in)
        cols_in_fact = df_fact_in.columns
        n_in_fact = df_fact_in.count()

        df_fact_out, exec_log_fact = apply_rules_for_table(
            table_key="cotacao_generico",
            df=df_fact_in,
            rules=RULES_BY_TABLE["cotacao_generico"],
            enable_rules=ENABLE_RULES,
        )
        rules_execution_log["cotacao_generico"] = exec_log_fact

        cols_out_fact = df_fact_out.columns
        n_out_fact = df_fact_out.count()

        # FL_NOVO: reset linhas existentes antes de escrever novas
        primeira_carga_fact = not table_exists(SILVER_FACT_FQN)
        if not primeira_carga_fact:
            spark.sql(f"UPDATE {SILVER_FACT_FQN} SET FL_NOVO = FALSE WHERE FL_NOVO = TRUE")

        write_mode_fact = "overwrite" if primeira_carga_fact else "append"
        (df_fact_out.write
            .format("delta")
            .mode(write_mode_fact)
            .saveAsTable(SILVER_FACT_FQN))
        tables_written["cotacao_generico"] = SILVER_FACT_FQN

        # -------------------------
        # 2) corretor_resumo — sobrescreve (tabela de referência, sem FL_NOVO)
        # -------------------------
        df_res_in = spark.table(BRONZE_CORRETOR_RESUMO_FQN)
        df_res_in = auto_cast_numeric_cols(df_res_in)
        cols_in_res = df_res_in.columns
        n_in_res = df_res_in.count()

        df_res_out, exec_log_res = apply_rules_for_table(
            table_key="corretor_resumo",
            df=df_res_in,
            rules=RULES_BY_TABLE["corretor_resumo"],
            enable_rules=ENABLE_RULES,
        )
        rules_execution_log["corretor_resumo"] = exec_log_res

        (df_res_out.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(SILVER_CORRETOR_RESUMO_FQN))
        tables_written["corretor_resumo"] = SILVER_CORRETOR_RESUMO_FQN

        cols_out_res = df_res_out.columns
        n_out_res = df_res_out.count()

        # -------------------------
        # 3) corretor_detalhe — sobrescreve (tabela de referência, sem FL_NOVO)
        # -------------------------
        df_det_in = spark.table(BRONZE_CORRETOR_DETALHE_FQN)
        df_det_in = auto_cast_numeric_cols(df_det_in)
        cols_in_det = df_det_in.columns
        n_in_det = df_det_in.count()

        df_det_out, exec_log_det = apply_rules_for_table(
            table_key="corretor_detalhe",
            df=df_det_in,
            rules=RULES_BY_TABLE["corretor_detalhe"],
            enable_rules=ENABLE_RULES,
        )
        rules_execution_log["corretor_detalhe"] = exec_log_det

        (df_det_out.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(SILVER_CORRETOR_DETALHE_FQN))
        tables_written["corretor_detalhe"] = SILVER_CORRETOR_DETALHE_FQN

        cols_out_det = df_det_out.columns
        n_out_det = df_det_out.count()

        # -------------------------
        # Metrics de regras
        # -------------------------
        for tbl, logs in rules_execution_log.items():
            for item in logs:
                if item["status"] == "APPLIED":
                    n_rules_applied += 1
                else:
                    n_rules_skipped += 1

        mlflow.log_metric("n_tabelas_processadas", 3)
        mlflow.log_metric("n_rules_total",         int(n_rules_total))
        mlflow.log_metric("n_rules_applied",       int(n_rules_applied))
        mlflow.log_metric("n_rules_skipped",       int(n_rules_skipped))
        mlflow.log_metric("n_linhas_bronze_novos", int(n_in_fact))
        mlflow.log_metric("n_linhas_silver_novos", int(n_out_fact))

        mlflow.log_dict(rules_execution_log, "rules_execution.json")

        # -------------------------
        # Profiling
        # -------------------------
        profiling = {
            "cotacao_generico": {
                "input":  {"total_linhas": int(n_in_fact),  "total_colunas": len(cols_in_fact),  "colunas": list(cols_in_fact)},
                "output": {"total_linhas": int(n_out_fact), "total_colunas": len(cols_out_fact), "colunas": list(cols_out_fact)},
            },
            "corretor_resumo": {
                "input":  {"total_linhas": int(n_in_res),  "total_colunas": len(cols_in_res),  "colunas": list(cols_in_res)},
                "output": {"total_linhas": int(n_out_res), "total_colunas": len(cols_out_res), "colunas": list(cols_out_res)},
            },
            "corretor_detalhe": {
                "input":  {"total_linhas": int(n_in_det),  "total_colunas": len(cols_in_det),  "colunas": list(cols_in_det)},
                "output": {"total_linhas": int(n_out_det), "total_colunas": len(cols_out_det), "colunas": list(cols_out_det)},
            },
        }
        mlflow.log_dict(profiling, "profiling/profiling.json")

print("PRE_PROC_INF concluído")
print("• silver fato:", SILVER_FACT_FQN)
print("• silver corretor_resumo:", SILVER_CORRETOR_RESUMO_FQN)
print("• silver corretor_detalhe:", SILVER_CORRETOR_DETALHE_FQN)
print(f"• EXEC_RUN_ID: {cr.info.run_id}")


# COMMAND ----------

