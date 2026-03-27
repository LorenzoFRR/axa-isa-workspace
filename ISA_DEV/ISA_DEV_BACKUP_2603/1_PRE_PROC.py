# Databricks notebook source
# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import json
from typing import Callable, Dict, List, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from functools import reduce

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cria schema

# COMMAND ----------

schema_name = "silver"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configs

# COMMAND ----------

# =========================
# MLflow
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"
PARENT_RUN_NAME = "T_PR_PRE_PROC"   # parent run só para acomodar a CR

# =========================
# Versão/timestamp da etapa
# =========================
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")

# =========================
# Inputs (bronze) — DEFINIR MANUALMENTE
# =========================
# Fato: você aponta manualmente qual tabela versionada quer usar
BRONZE_FACT_FQN = "bronze.cotacao_generico_20260318_110703"  # <<< AJUSTE

# Dimensões (nome fixo no bronze)
BRONZE_CORRETOR_RESUMO_FQN  = "bronze.corretor_resumo_carga_1703"
BRONZE_CORRETOR_DETALHE_FQN = "bronze.corretor_detalhe_carga_1703"

# =========================
# Outputs (silver) — versionados por execução
# =========================
SILVER_SCHEMA = "silver"

SILVER_FACT_FQN = f"{SILVER_SCHEMA}.cotacao_generico_clean_{TS_EXEC}"
SILVER_CORRETOR_RESUMO_FQN  = f"{SILVER_SCHEMA}.corretor_resumo_clean_{TS_EXEC}"
SILVER_CORRETOR_DETALHE_FQN = f"{SILVER_SCHEMA}.corretor_detalhe_clean_{TS_EXEC}"

# Escrita (como o nome é único, overwrite é seguro)
WRITE_MODE = "overwrite"  # overwrite recomendado

# =========================
# Flags para desenvolvimento dinâmico de regras
# =========================
# Interruptor global: False desabilita TODAS as regras de TODAS as tabelas
ENABLE_RULES = True

# =========================
# Acoplar run pai existente (opcional)
# =========================
# Preencha com o run_id do T_PR_PRE_PROC existente para não criar outro container.
# Deixe vazio ("") para criar uma nova parent run.
PR_RUN_ID_OVERRIDE = "b7930f5d5b9a452e895346e2d53a7017"

# =========================
# Colunas técnicas (útil para regras e drops)
# =========================
TECH_COLS = ["TS_ARQ", "TS_ATUALIZACAO", "SOURCE_FILE"]

print("✅ CONFIG 1_PRE_PROC carregada")
print("• bronze fact:", BRONZE_FACT_FQN)
print("• silver fact:", SILVER_FACT_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define helpers

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
        raise ValueError(f"❌ Tabela não existe: {fqn}")


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
    # NULL se string vazia / só espaços
    return F.when(F.trim(col) == F.lit(""), F.lit(None)).otherwise(col)

def normalize_ptbr_number_str(col: F.Column) -> F.Column:
    """
    Normaliza string numérica estilo PT-BR:
    - remove aspas e espaços
    - remove separador de milhar "." quando existir
    - troca "," por "."
    """
    c = F.regexp_replace(col.cast("string"), r'["\s]', "")
    c = F.regexp_replace(c, r"\.", "")   # remove milhares
    c = F.regexp_replace(c, r",", ".")   # decimal
    return c

def cast_decimal(df: DataFrame, col_name: str, precision: int, scale: int) -> DataFrame:
    if col_name not in df.columns:
        return df
    dec_type = T.DecimalType(precision=precision, scale=scale)
    return df.withColumn(col_name, normalize_ptbr_number_str(F.col(col_name)).cast(dec_type))

def drop_all_null_or_blank_columns(df: DataFrame) -> DataFrame:
    """
    Remove colunas 100% vazias (NULL ou ''), útil p/ corretor_resumo.
    Custo: faz um agg para cada coluna (aceitável para tabelas pequenas; se ficar pesado, refinamos).
    """
    exprs = []
    for c, t in df.dtypes:
        if t == "string":
            exprs.append(F.max(F.when(F.trim(F.col(c)) != "", F.lit(1)).otherwise(F.lit(0))).alias(c))
        else:
            exprs.append(F.max(F.when(F.col(c).isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(c))

    flags = df.agg(*exprs).collect()[0].asDict()
    keep = [c for c in df.columns if flags.get(c, 1) == 1]  # default keep
    return df.select(*keep)

def auto_cast_numeric_cols(df: DataFrame) -> DataFrame:
    """
    Detecta e faz cast automático de colunas STRING que contêm apenas
    valores numéricos (inteiros ou decimais). Uma única ação Spark.
    Colunas com conteúdo textual real são mantidas como STRING.
    Tenta long primeiro; se falhar, tenta double.
    """
    string_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "string"]
    if not string_cols:
        return df

    agg_exprs = []
    for c in string_cols:
        non_null = F.col(c).isNotNull() & (F.trim(F.col(c)) != "")
        agg_exprs += [
            F.count(F.when(non_null & ~F.col(c).rlike(r"^-?\d+$"),       True)).alias(f"{c}__long_fail"),
            F.count(F.when(non_null & F.col(c).cast("double").isNull(), True)).alias(f"{c}__dbl_fail"),
        ]

    stats = df.agg(*agg_exprs).collect()[0].asDict()

    for c in string_cols:
        if stats[f"{c}__long_fail"] == 0:
            df = df.withColumn(c, F.col(c).cast("long"))
        elif stats[f"{c}__dbl_fail"] == 0:
            df = df.withColumn(c, F.col(c).cast("double"))

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
        "fn": fn,  # não será logado; só para execução
    }

def _is_rule_enabled(table_key: str, rule_id: str, base_enabled: bool, toggles: Dict) -> bool:
    """
    toggles = { "cotacao_generico": {"GEN_R06": False}, ... }
    """
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
    """
    Aplica regras em ordem e retorna:
    - df_out
    - exec_log: lista de {rule_id, status, description, reason?}
    Status: APPLIED | SKIPPED_DISABLED | SKIPPED_MISSING_COLS
    """
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

        # aplica
        try:
            df_out = r["fn"](df_out)
            exec_log.append({"rule_id": rid, "status": "APPLIED", "description": desc})
        except Exception as e:
            # Falha de regra deve falhar o run (para não gerar silver inconsistente)
            raise RuntimeError(f"❌ Falha ao aplicar regra {table_key}.{rid}: {desc}. Erro: {e}") from e

    return df_out, exec_log


# =========================
# Helpers: logging (catálogo, execução, lineage)
# =========================
def build_rules_catalog_for_logging(rules_by_table: Dict[str, List[Dict]]) -> Dict:
    """
    Remove o campo 'fn' para não logar código/funções; mantém ids/descrições/enabled/requires_columns.
    """
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
    """
    Referência explícita bronze -> silver (manual no input).
    """
    return {
        "cotacao_generico": {"bronze": BRONZE_FACT_FQN, "silver": SILVER_FACT_FQN},
        "corretor_resumo":  {"bronze": BRONZE_CORRETOR_RESUMO_FQN, "silver": SILVER_CORRETOR_RESUMO_FQN},
        "corretor_detalhe": {"bronze": BRONZE_CORRETOR_DETALHE_FQN, "silver": SILVER_CORRETOR_DETALHE_FQN},
        "ts_exec": TS_EXEC,
    }


print("✅ CEL 1 carregada: helpers + executor de regras prontos")



# =========================================================
# Helpers específicos (iguais ao notebook V6)
# =========================================================
def clean_numeric_str_us(colname: str) -> F.Column:
    """
    Igual ao notebook:
    - trim
    - ""/NULL -> NULL
    - remove separador de milhar ","
    - ".02" -> "0.02" | "-.02" -> "-0.02"
    """
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
    """
    Igual ao notebook V6 (robusto para múltiplos formatos).
    """
    x = F.col(cname).cast("string")
    return F.coalesce(
        F.to_date(F.col(cname)),
        F.to_date(F.to_timestamp(x, "yyyy-MM-dd'T'HH:mm:ss.SSSXXX")),
        F.to_date(F.to_timestamp(x, "yyyy-MM-dd'T'HH:mm:ssXXX")),
        F.to_date(F.to_timestamp(x, "yyyy-MM-dd HH:mm:ss.SSS")),
        F.to_date(F.to_timestamp(x, "yyyy-MM-dd HH:mm:ss")),
        F.to_date(F.to_timestamp(x, "yyyy-MM-dd")),
        F.to_date(F.to_timestamp(x, "dd/MM/yyyy HH:mm:ss")),
        F.to_date(F.to_timestamp(x, "dd/MM/yyyy")),
        F.to_date(x),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define regras de negócio e pré-processamento

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regras COTACAO_GENERICO

# COMMAND ----------

# =========================================================
# Regras: cotacao_generico (baseado no notebook V6)
# =========================================================
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
    fl_raw = F.col("FL_ANALISE_SUBSCRICAO")
    fl_manual = (
        (fl_raw == F.lit(True)) |
        (F.lower(F.trim(fl_raw.cast("string"))).isin("sim", "true", "1"))
    )
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

    # mantém último snapshot por cotação
    w = Window.partitionBy("CD_NUMERO_COTACAO_AXA").orderBy(F.col("TS_ARQ_ts").desc_nulls_last())
    df_latest = df1.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

    # referência (máximo global do TS_ARQ no df_latest)
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

    # map: DT_* -> DIAS_*
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

    # DATA_COTACAO parse + DIAS_COTACAO
    if "DATA_COTACAO" in df_out2.columns:
        df_out2 = df_out2.withColumn("DATA_COTACAO", parse_to_date_any("DATA_COTACAO"))
        df_out2 = df_out2.withColumn("DIAS_COTACAO", F.datediff(F.col("DATA_COTACAO"), F.col("REF_DATE")).cast("int"))

    # DIAS_INICIO_VIGENCIA = DT_INICIO_VIGENCIA - DATA_COTACAO
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
    # Normaliza casing de DS_GRUPO_STATUS para os valores canônicos do modelo.
    # Movida de PP_R01 (3_ML_TREINO_MODE_B_1).
    s = F.upper(F.trim(F.col("DS_GRUPO_STATUS").cast("string")))
    return df.withColumn(
        "DS_GRUPO_STATUS",
        F.when(s == "EMITIDA", F.lit("Emitida"))
         .when(s == "PERDIDA", F.lit("Perdida"))
         .otherwise(F.col("DS_GRUPO_STATUS"))
    )

def GEN_R10_filtra_status_finais(df: DataFrame) -> DataFrame:
    # Mantém apenas cotações com desfecho conhecido (Emitida ou Perdida).
    # Pipeline de treino: cotações intermediárias não possuem label definido.
    return df.filter(F.col("DS_GRUPO_STATUS").isin(["Emitida", "Perdida"]))

def GEN_R11_cria_label(df: DataFrame) -> DataFrame:
    # Cria coluna `label`: Emitida=1, Perdida=0, demais=NULL.
    # Movida de PP_R03 (3_ML_TREINO_MODE_B_1).
    return df.withColumn(
        "label",
        F.when(F.col("DS_GRUPO_STATUS") == "Emitida", F.lit(1))
         .when(F.col("DS_GRUPO_STATUS") == "Perdida", F.lit(0))
         .otherwise(F.lit(None).cast("int"))
    )

def GEN_R12_cria_mes(df: DataFrame) -> DataFrame:
    # Cria coluna MES=yyyy-MM a partir de DATA_COTACAO; remove linhas com DATA_COTACAO nula.
    # Movida de BUILD_R01 (3_ML_TREINO_MODE_B_1).
    return (
        df.withColumn("DATA_COTACAO_dt", F.to_date(F.col("DATA_COTACAO")))
          .filter(F.col("DATA_COTACAO_dt").isNotNull())
          .withColumn("MES", F.date_format("DATA_COTACAO_dt", "yyyy-MM"))
          .drop("DATA_COTACAO_dt")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regras CORRETOR_RESUMO

# COMMAND ----------

# =========================================================
# Regras: corretor_resumo (baseado no notebook V6)
# =========================================================
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
    # Colunas do corretor_resumo sem utilidade downstream.
    # Movida do JOIN_R03 (2_JOIN) para centralizar aqui.
    # Nota: DS_CANAL_COMERCIAL é normalizada por RES_R03 e depois removida aqui;
    # considere desabilitar RES_R03 se a normalização não for mais necessária.
    cols = [
        "DS_CORRETOR",
        "DS_SEGMENTACAO",
        "DS_SUCURSAL_AUTORIZA_EMISSAO",
        "DS_SEGMENTACAO_GRUPO",
        "DS_CANAL_COMERCIAL",
    ]
    return safe_drop_cols(df, cols)

def RES_R07_drop_colunas_flag(df: DataFrame) -> DataFrame:
    # Flags do corretor_resumo sem utilidade downstream.
    # Movida do JOIN_R04 (2_JOIN) para centralizar aqui.
    cols = [
        "FL_PERMITE_ANTECIPAR",
        "FL_ACORDO_GENERALISTA",
        "FL_ACORDO_ESPECIFICO",
    ]
    return safe_drop_cols(df, cols)

def RES_R08_uppercase_cols(df: DataFrame) -> DataFrame:
    # Garante que todos os nomes de coluna do corretor_resumo estejam em UPPERCASE
    # antes de chegarem ao JOIN. Movida do JOIN_R05 (2_JOIN) para centralizar aqui.
    return df.toDF(*[c.upper() for c in df.columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regras CORRETOR_DETALHE

# COMMAND ----------

# =========================================================
# Regras: corretor_detalhe (baseado no notebook V6)
# =========================================================
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
# MAGIC ### Catálogo dinâmico

# COMMAND ----------

# =========================================================
# Catálogo dinâmico (RULES_BY_TABLE)
# Toggles inline: True = regra ativa | False = regra desabilitada
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
        rule_def("GEN_R08", "Dropar colunas fora do escopo (lista do notebook V6)",
                 GEN_R08_drop_colunas, enabled=True),
        rule_def("GEN_R09", "Normalizar casing de DS_GRUPO_STATUS para valores canônicos (Emitida, Perdida)",
                 GEN_R09_normaliza_status, enabled=True,
                 requires_columns=["DS_GRUPO_STATUS"]),
        rule_def("GEN_R10", "Filtrar apenas cotações com desfecho conhecido (Emitida ou Perdida)",
                 GEN_R10_filtra_status_finais, enabled=True,
                 requires_columns=["DS_GRUPO_STATUS"]),
        rule_def("GEN_R11", "Criar coluna label (Emitida=1, Perdida=0) como INT",
                 GEN_R11_cria_label, enabled=True,
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
        rule_def("RES_R04", "Dropar colunas técnicas e colunas específicas (lista do notebook V6)",
                 RES_R04_drop_cols_tecnicas, enabled=True),
        rule_def("RES_R05", "Dropar colunas 100% vazias (NULL ou '')", RES_R05_drop_cols_100pct_vazias,
                 enabled=True),
        rule_def("RES_R06", "Dropar colunas extras sem utilidade downstream (DS_CORRETOR, DS_SEGMENTACAO, DS_SUCURSAL_AUTORIZA_EMISSAO, DS_SEGMENTACAO_GRUPO, DS_CANAL_COMERCIAL)",
                 RES_R06_drop_colunas_extras, enabled=True),
        rule_def("RES_R07", "Dropar colunas de flag sem utilidade downstream (FL_PERMITE_ANTECIPAR, FL_ACORDO_GENERALISTA, FL_ACORDO_ESPECIFICO)",
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

print("✅ CEL 2 carregada: RULES_BY_TABLE definido")
print("• regras cotacao_generico:", len(RULES_BY_TABLE["cotacao_generico"]))
print("• regras corretor_resumo:", len(RULES_BY_TABLE["corretor_resumo"]))
print("• regras corretor_detalhe:", len(RULES_BY_TABLE["corretor_detalhe"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executa operações, loga MLflow e cria tabelas em silver

# COMMAND ----------

# =========================
# Validações e bootstrap de schemas
# =========================
ensure_schema(SILVER_SCHEMA)

assert_table_exists(BRONZE_FACT_FQN)
assert_table_exists(BRONZE_CORRETOR_RESUMO_FQN)
assert_table_exists(BRONZE_CORRETOR_DETALHE_FQN)

if table_exists(SILVER_FACT_FQN):
    raise ValueError(f"❌ Silver fato já existe: {SILVER_FACT_FQN} (timestamp repetido?)")
if table_exists(SILVER_CORRETOR_RESUMO_FQN):
    raise ValueError(f"❌ Silver corretor_resumo já existe: {SILVER_CORRETOR_RESUMO_FQN} (timestamp repetido?)")
if table_exists(SILVER_CORRETOR_DETALHE_FQN):
    raise ValueError(f"❌ Silver corretor_detalhe já existe: {SILVER_CORRETOR_DETALHE_FQN} (timestamp repetido?)")

# =========================
# MLflow bootstrap
# =========================
_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

child_run_name = f"T_PRE_PROC_{TS_EXEC}"

rules_catalog_for_logging = build_rules_catalog_for_logging(RULES_BY_TABLE)
tables_lineage = build_tables_lineage_dict()

# Vai acumular logs por tabela
rules_execution_log: Dict[str, List[Dict]] = {}
tables_written: Dict[str, str] = {}

# Contadores (metrics)
n_rules_total = sum(len(v) for v in RULES_BY_TABLE.values())
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
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "PRE_PROC")
        mlflow.set_tag("run_role", "parent")

    with mlflow.start_run(run_name=child_run_name, nested=True) as cr:
        # CR tags
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "PRE_PROC")
        mlflow.set_tag("run_role", "child")

        # CR params: inputs bronze
        mlflow.log_param("bronze_fact_fqn", BRONZE_FACT_FQN)
        mlflow.log_param("bronze_corretor_resumo_fqn", BRONZE_CORRETOR_RESUMO_FQN)
        mlflow.log_param("bronze_corretor_detalhe_fqn", BRONZE_CORRETOR_DETALHE_FQN)

        # CR params: outputs silver
        mlflow.log_param("silver_fact_fqn", SILVER_FACT_FQN)
        mlflow.log_param("silver_corretor_resumo_fqn", SILVER_CORRETOR_RESUMO_FQN)
        mlflow.log_param("silver_corretor_detalhe_fqn", SILVER_CORRETOR_DETALHE_FQN)

        # CR params: config do run
        mlflow.log_param("ts_exec", TS_EXEC)
        mlflow.log_param("write_mode", WRITE_MODE)
        mlflow.log_param("enable_rules", str(ENABLE_RULES))

        # Artefatos fixos: catálogo + lineage (antes de executar)
        mlflow.log_dict(rules_catalog_for_logging, "rules_catalog.json")
        mlflow.log_dict(tables_lineage, "tables_lineage.json")

        # -------------------------
        # 1) cotacao_generico
        # -------------------------
        df_fact_in = spark.table(BRONZE_FACT_FQN)
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

        (df_fact_out.write
            .format("delta")
            .mode(WRITE_MODE)
            .saveAsTable(SILVER_FACT_FQN))
        tables_written["cotacao_generico"] = SILVER_FACT_FQN

        cols_out_fact = df_fact_out.columns
        n_out_fact = df_fact_out.count()

        # -------------------------
        # 2) corretor_resumo
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
            .mode(WRITE_MODE)
            .saveAsTable(SILVER_CORRETOR_RESUMO_FQN))
        tables_written["corretor_resumo"] = SILVER_CORRETOR_RESUMO_FQN

        cols_out_res = df_res_out.columns
        n_out_res = df_res_out.count()

        # -------------------------
        # 3) corretor_detalhe
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
            .mode(WRITE_MODE)
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
        mlflow.log_metric("n_rules_total", int(n_rules_total))
        mlflow.log_metric("n_rules_applied", int(n_rules_applied))
        mlflow.log_metric("n_rules_skipped", int(n_rules_skipped))

        # Artefato com o detalhamento da execução das regras
        mlflow.log_dict(rules_execution_log, "rules_execution.json")

        # -------------------------
        # Profiling: totais de linhas/colunas por tabela
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

print("PRE_PROC concluído")
print("• silver fato:", SILVER_FACT_FQN)
print("• silver corretor_resumo:", SILVER_CORRETOR_RESUMO_FQN)
print("• silver corretor_detalhe:", SILVER_CORRETOR_DETALHE_FQN)