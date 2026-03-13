# Databricks notebook source
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configs

# COMMAND ----------

# =========================
# MLflow
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"      # <<< AJUSTE
PARENT_RUN_NAME = "T_PR_JOIN"   # parent run só para acomodar a CR

# =========================
# Versão/timestamp da etapa
# =========================
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")

# =========================
# Inputs (silver) — DEFINIR MANUALMENTE
# =========================
SILVER_FACT_FQN   = "silver.cotacao_generico_clean_20260310_163104"      # <<< AJUSTE
SILVER_RESUMO_FQN = "silver.corretor_resumo_clean_20260310_163104"       # <<< AJUSTE
SILVER_DETALHE_FQN= "silver.corretor_detalhe_clean_20260310_163104"      # <<< AJUSTE

# =========================
# Outputs (silver) — versionados por execução
# =========================
OUT_SCHEMA = "silver"

# tabela final com segmentação (SEG)
SEG_TABLE_FQN  = f"{OUT_SCHEMA}.cotacao_seg_{TS_EXEC}"

WRITE_MODE = "overwrite"  # como o nome é único, overwrite é seguro

# =========================
# Regras dinâmicas
# =========================
ENABLE_RULES = True

# Permite desligar regras específicas sem apagar código:
# Ex.: {"cotacao_join": {"JOIN_R02": False}, "cotacao_seg": {"SEG_R02": False}}
RULE_TOGGLES = {}

# =========================
# Acoplar run pai existente (opcional)
# =========================
# Preencha com o run_id do T_PR_JOIN existente para não criar outro container.
# Deixe vazio ("") para criar uma nova parent run.
PR_RUN_ID_OVERRIDE = "77362347d05443b18abc9a71dc2b3ff1"

# drops após criar SEG
DROP_COLS_AFTER_SEG = ["CANAL", "DS_TIPO_COTACAO"]

# =========================
# Profiling / artifacts esperados
# =========================
DO_PROFILE = True  # aqui vamos gerar profiling (seg)

SEG_COL = "SEG"
STATUS_COL = "DS_GRUPO_STATUS"

print("✅ CONFIG 2_JOIN carregada")
print("• inputs:", SILVER_FACT_FQN, SILVER_RESUMO_FQN, SILVER_DETALHE_FQN)
print("• seg  out:", SEG_TABLE_FQN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define helpers

# COMMAND ----------

# =========================
# Metastore helpers
# =========================
import os
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


# =========================
# DataFrame helpers
# =========================
def safe_drop_cols(df: DataFrame, cols: List[str]) -> DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(*existing) if existing else df

def normalize_key(col: F.Column) -> F.Column:
    """
    Normalização padrão de chaves (como no notebook V6):
    trim + "" -> NULL
    """
    c = F.trim(col.cast("string"))
    return F.when(c.isNull() | (c == F.lit("")), F.lit(None)).otherwise(c)

def uppercase_columns(df: DataFrame) -> DataFrame:
    # mantém ordem; renomeia colunas para uppercase
    return df.toDF(*[c.upper() for c in df.columns])

def with_uppercase_values(df: DataFrame, cols: List[str]) -> DataFrame:
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.upper(F.col(c).cast("string")))
    return df


# =========================
# Executor de transformações (2 fases)
# =========================
TransformFn = Callable[[DataFrame], DataFrame]

def transform_def(
    transform_id: str,
    description: str,
    fn: TransformFn,
    enabled: bool = True,
    requires_columns: Optional[List[str]] = None,
) -> Dict:
    return {
        "transform_id": transform_id,
        "description": description,
        "enabled": enabled,
        "requires_columns": requires_columns or [],
        "fn": fn,
    }

def _is_enabled(phase_key: str, transform_id: str, base_enabled: bool, toggles: Dict) -> bool:
    """
    toggles = { "post_join": {"JOIN_A02": False}, "post_seg": {"SEG_B01": False} }
    """
    if not base_enabled:
        return False
    if not toggles:
        return True
    t = toggles.get(phase_key, {})
    if transform_id in t:
        return bool(t[transform_id])
    return True

def apply_transforms(
    phase_key: str,
    df: DataFrame,
    transforms: List[Dict],
    enable_rules: bool = True,
    toggles: Optional[Dict] = None,
) -> Tuple[DataFrame, List[Dict]]:
    """
    Retorna df_out e exec_log:
      {transform_id, status, description, reason?}
    status: APPLIED | SKIPPED_DISABLED | SKIPPED_MISSING_COLS
    """
    toggles = toggles or {}
    exec_log: List[Dict] = []
    df_out = df

    for t in transforms:
        tid = t["transform_id"]
        desc = t["description"]
        req = t.get("requires_columns", []) or []

        enabled = _is_enabled(phase_key, tid, t.get("enabled", True), toggles) and enable_rules
        if not enabled:
            exec_log.append({"transform_id": tid, "status": "SKIPPED_DISABLED", "description": desc})
            continue

        missing = [c for c in req if c not in df_out.columns]
        if missing:
            exec_log.append({
                "transform_id": tid,
                "status": "SKIPPED_MISSING_COLS",
                "description": desc,
                "reason": f"missing_columns={missing}",
            })
            continue

        try:
            df_out = t["fn"](df_out)
            exec_log.append({"transform_id": tid, "status": "APPLIED", "description": desc})
        except Exception as e:
            raise RuntimeError(f"❌ Falha transform {phase_key}.{tid}: {desc}. Erro: {e}") from e

    return df_out, exec_log

def transforms_catalog_for_logging(transforms_by_phase: Dict[str, List[Dict]]) -> Dict:
    out = {}
    for phase_key, transforms in transforms_by_phase.items():
        out[phase_key] = [
            {
                "transform_id": t["transform_id"],
                "description": t["description"],
                "enabled": bool(t.get("enabled", True)),
                "requires_columns": t.get("requires_columns", []) or [],
            }
            for t in transforms
        ]
    return out


# =========================
# Profiling (simples, objetivo)
# =========================
def profile_basic(df: DataFrame, name: str, key_cols: Optional[List[str]] = None) -> Dict:
    """
    Profiling leve:
      - n_rows, n_cols
      - null_count por coluna (amostra completa, via agg)
      - distinct_count para key_cols (se fornecido)
    """
    n_rows = df.count()
    n_cols = len(df.columns)

    # nulls por coluna (considera "" como nulo para strings)
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

def seg_counts(df: DataFrame, seg_col: str) -> List[Dict]:
    if seg_col not in df.columns:
        return []
    rows = (df.groupBy(seg_col)
              .count()
              .orderBy(F.col("count").desc())
              .collect())
    return [{seg_col: r[seg_col], "count": int(r["count"])} for r in rows]

def status_dist_by_seg(df: DataFrame, seg_col: str, status_col: str) -> List[Dict]:
    if seg_col not in df.columns or status_col not in df.columns:
        return []
    # counts por (seg, status)
    base = (df.groupBy(seg_col, status_col).count())
    # total por seg
    tot = df.groupBy(seg_col).count().withColumnRenamed("count", "seg_total")
    # join e proporção
    out = (base.join(tot, on=seg_col, how="left")
               .withColumn("pct", (F.col("count") / F.col("seg_total")).cast("double"))
               .orderBy(F.col(seg_col), F.col("count").desc()))
    rows = out.collect()
    return [
        {seg_col: r[seg_col], status_col: r[status_col], "count": int(r["count"]), "seg_total": int(r["seg_total"]), "pct": float(r["pct"])}
        for r in rows
    ]


# =========================
# Artifacts adicionais: evolução mensal por DS_GRUPO_STATUS (DATA_COTACAO)
# =========================
def log_status_by_month_artifacts(
    df_seg: DataFrame,
    *,
    date_col: str = "DATA_COTACAO",
    status_col: str = "DS_GRUPO_STATUS",
    artifact_prefix: str = "analysis",
    source_table_fqn: Optional[str] = None,
) -> None:
    """
    Gera e loga como artifacts:
      - pivot mensal (CSV + JSON)
      - gráfico barras empilhadas (PNG)

    Espera ser chamado DENTRO de uma mlflow run ativa.
    """
    missing = [c for c in [date_col, status_col] if c not in df_seg.columns]
    if missing:
        mlflow.log_dict(
            {"ok": False, "reason": "missing_columns", "missing": missing, "source_table": source_table_fqn},
            f"{artifact_prefix}/status_by_month_info.json",
        )
        return

    df_m = (
        df_seg
        .withColumn("DATA_COTACAO_dt", F.to_date(F.col(date_col)))
        .filter(F.col("DATA_COTACAO_dt").isNotNull())
        .withColumn("MES", F.date_format(F.col("DATA_COTACAO_dt"), "yyyy-MM"))
        .withColumn("STATUS", F.upper(F.trim(F.col(status_col).cast("string"))))
    )

    if df_m.rdd.isEmpty():
        mlflow.log_dict(
            {"ok": False, "reason": "no_rows_after_date_filter", "source_table": source_table_fqn},
            f"{artifact_prefix}/status_by_month_info.json",
        )
        return

    df_pivot = (
        df_m.groupBy("MES")
            .pivot("STATUS")
            .agg(F.count(F.lit(1)))
            .orderBy("MES")
    )

    # toPandas para gerar artefatos (pivot costuma ser pequeno: meses x status)
    pdf = df_pivot.toPandas().fillna(0)

    if pdf.empty:
        mlflow.log_dict(
            {"ok": False, "reason": "empty_pivot", "source_table": source_table_fqn},
            f"{artifact_prefix}/status_by_month_info.json",
        )
        return

    status_cols = [c for c in pdf.columns if c != "MES"]
    for c in status_cols:
        pdf[c] = pdf[c].astype(int)

    # --------- artifacts estruturados ----------
    mlflow.log_text(pdf.to_csv(index=False), f"{artifact_prefix}/status_by_month_pivot.csv")

    pivot_json = {
        "ok": True,
        "source_table": source_table_fqn,
        "n_months": int(len(pdf)),
        "months": pdf["MES"].tolist(),
        "status_cols": status_cols,
        "rows": pdf.to_dict(orient="records"),
    }
    mlflow.log_dict(pivot_json, f"{artifact_prefix}/status_by_month_pivot.json")

    # --------- gráfico PNG ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uuid

    fig = plt.figure(figsize=(12, 5))
    bottom = None
    x = range(len(pdf["MES"]))

    for col in status_cols:
        if bottom is None:
            plt.bar(x, pdf[col], label=col)
            bottom = pdf[col].copy()
        else:
            plt.bar(x, pdf[col], bottom=bottom, label=col)
            bottom = bottom + pdf[col]

    plt.xticks(list(x), pdf["MES"], rotation=45, ha="right")
    plt.xlabel("Mês (YYYY-MM)")
    plt.ylabel("Quantidade de linhas")
    plt.title("Total de linhas por mês (DATA_COTACAO), estratificado por DS_GRUPO_STATUS")
    plt.grid(False, axis="y")
    plt.legend()
    plt.tight_layout()

    png_path = f"/tmp/status_by_month_stacked_{uuid.uuid4().hex}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(png_path, artifact_path=artifact_prefix)


print("✅ Helpers carregados: executor + profiling + artifacts (status_by_month) prontos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lineage e Join

# COMMAND ----------

# =========================
# Join spec + lineage (para artifacts)
# =========================
tables_lineage = {
    "stage": "2_JOIN",
    "ts_exec": TS_EXEC,
    "inputs": {
        "fact": SILVER_FACT_FQN,
        "corretor_resumo": SILVER_RESUMO_FQN,
        "corretor_detalhe": SILVER_DETALHE_FQN,
    },
    "outputs": {
        "seg_table": SEG_TABLE_FQN,
    }
}

join_spec = {
    "join_1": {
        "left": "fact",
        "right": "corretor_resumo",
        "left_key": "CD_DOC_CORRETOR",
        "right_key": "CD_CORRETOR",
        "right_payload_suffix": "_resumo",
        "retention_rule": "keep if left_key is NULL/blank OR match exists in resumo",
    },
    "join_2": {
        "left": "join_1_out",
        "right": "corretor_detalhe",
        "left_keys": ["CD_DOC_CORRETOR", "DS_PRODUTO_NOME", "DS_TIPO_COTACAO"],
        "right_keys": ["CD_DOC_CORRETOR", "DS_PRODUTO_NOME", "DS_TIPO_SOLICITACAO"],
        "right_payload_suffix": "_detalhe",
        "right_filter": "only rows with all 3 right keys not null",
        "right_dedup": "dropDuplicates on 3 right keys",
        "retention_rule": "keep if CD_DOC_CORRETOR is NULL/blank OR (doc+produto+tipo present AND match exists in detalhe)",
    },
    "seg": {
        "seg_col": SEG_COL,
        "from_cols": ["CANAL", "DS_TIPO_COTACAO"],
        "canal_map": {"digital": "DIGITAL", "manual": "MANUAL"},
        "tipo_map": {"seguro novo": "SEGURO_NOVO", "renovação axa|renovacao axa": "RENOVACAO"},
        "rule": "SEG = TIPO_CANAL when both not null, else NULL",
        "drop_after": DROP_COLS_AFTER_SEG,
    }
}


# =========================
# Transform functions (JOIN_1 / JOIN_2 iguais ao notebook)
# =========================
def _join_1_gen_resumo(df_gen: DataFrame, df_resumo: DataFrame) -> DataFrame:
    # 1) Normaliza chaves
    gen_n = df_gen.withColumn("key_gen", normalize_key(F.col("CD_DOC_CORRETOR")))
    res_n = df_resumo.withColumn("key_res", normalize_key(F.col("CD_CORRETOR")))

    # 2) Renomeia payload do resumo com sufixo _resumo (exceto CD_CORRETOR)
    res_payload_cols = [c for c in df_resumo.columns if c != "CD_CORRETOR"]
    res_renamed = res_n.select(
        F.col("key_res"),
        *[F.col(c).alias(f"{c}_resumo") for c in res_payload_cols]
    )

    # 3) Left join + retenção:
    # - mantém quando key_gen é NULL
    # - mantém quando key_gen preenchida SOMENTE se houver match (key_res not null)
    df_join_1 = (
        gen_n
        .join(res_renamed, gen_n["key_gen"] == res_renamed["key_res"], "left")
        .filter(F.col("key_gen").isNull() | F.col("key_res").isNotNull())
        .drop("key_gen", "key_res")
    )
    return df_join_1

def _join_2_join1_detalhe(df_join_1: DataFrame, df_detalhe: DataFrame) -> DataFrame:
    # 1) Normaliza chaves no df_join_1
    gen_n = (
        df_join_1
        .withColumn("cd_gen",   normalize_key(F.col("CD_DOC_CORRETOR")))
        .withColumn("prod_gen", normalize_key(F.col("DS_PRODUTO_NOME")))
        .withColumn("tipo_gen", normalize_key(F.col("DS_TIPO_COTACAO")))
    )

    # 2) Normaliza chaves no df_detalhe
    res_n = (
        df_detalhe
        .withColumn("cd_res",   normalize_key(F.col("CD_DOC_CORRETOR")))
        .withColumn("prod_res", normalize_key(F.col("DS_PRODUTO_NOME")))
        .withColumn("tipo_res", normalize_key(F.col("DS_TIPO_SOLICITACAO")))
    )

    # payload do detalhe (não traz as chaves originais)
    res_payload_cols = [
        c for c in df_detalhe.columns
        if c not in ["CD_DOC_CORRETOR", "DS_PRODUTO_NOME", "DS_TIPO_SOLICITACAO"]
    ]

    res_enrich = (
        res_n
        .filter(F.col("cd_res").isNotNull() & F.col("prod_res").isNotNull() & F.col("tipo_res").isNotNull())
        .select(
            "cd_res", "prod_res", "tipo_res",
            *[F.col(c).alias(f"{c}_detalhe") for c in res_payload_cols]
        )
        .dropDuplicates(["cd_res", "prod_res", "tipo_res"])
    )

    # 3) Join
    joined = gen_n.join(
        res_enrich,
        on=[
            gen_n["cd_gen"] == res_enrich["cd_res"],
            gen_n["prod_gen"] == res_enrich["prod_res"],
            gen_n["tipo_gen"] == res_enrich["tipo_res"],
        ],
        how="left",
    )

    # 4) Regra de retenção
    df_join_2 = (
        joined
        .withColumn("has_match", F.when(F.col("cd_res").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .filter(
            (F.col("cd_gen").isNull())
            |
            (
                F.col("cd_gen").isNotNull()
                & F.col("prod_gen").isNotNull()
                & F.col("tipo_gen").isNotNull()
                & (F.col("has_match") == 1)
            )
        )
        .drop("cd_gen", "prod_gen", "tipo_gen", "cd_res", "prod_res", "tipo_res", "has_match")
    )
    return df_join_2


# =========================
# SEG creation (igual ao notebook)
# =========================
def _seg_create(df: DataFrame) -> DataFrame:
    # after uppercase, columns are "CANAL" and "DS_TIPO_COTACAO"
    canal_norm = F.lower(F.trim(F.col("CANAL").cast("string")))
    canal_code = (
        F.when(canal_norm == "digital", F.lit("DIGITAL"))
         .when(canal_norm == "manual",  F.lit("MANUAL"))
         .otherwise(F.lit(None))
    )

    tipo_norm = F.lower(F.trim(F.col("DS_TIPO_COTACAO").cast("string")))
    tipo_code = (
        F.when(tipo_norm == "seguro novo", F.lit("SEGURO_NOVO"))
         .when(tipo_norm.isin("renovação axa", "renovacao axa"), F.lit("RENOVACAO"))
         .otherwise(F.lit(None))
    )

    return df.withColumn(
        SEG_COL,
        F.when(tipo_code.isNotNull() & canal_code.isNotNull(),
               F.concat_ws("_", tipo_code, canal_code)
        ).otherwise(F.lit(None))
    )

def _seg_drop_inputs(df: DataFrame) -> DataFrame:
    return safe_drop_cols(df, DROP_COLS_AFTER_SEG)


# =========================
# Catálogos (2 fases) — IDs + descrições logáveis
# =========================
# Fase 1: joins fact + resumo + detalhe.
# JOIN_R01 e JOIN_R02 dependem de dfs externos (resumo/detalhe),
# injetadas via closures na execução (CEL 3).
TRANSFORMS_COTACAO_JOIN = [
    {"transform_id": "JOIN_R01", "description": "Join fact + corretor_resumo com retenção (doc nulo OU match)", "enabled": True,
     "requires_columns": ["CD_DOC_CORRETOR"], "fn": None},
    {"transform_id": "JOIN_R02", "description": "Join resultado + corretor_detalhe (doc+produto+tipo) com retenção", "enabled": True,
     "requires_columns": ["CD_DOC_CORRETOR", "DS_PRODUTO_NOME", "DS_TIPO_COTACAO"], "fn": None},
]

# Fase 2: gera a tabela SEG (segmentação) + drops finais
TRANSFORMS_COTACAO_SEG = [
    transform_def("SEG_R01", "Criar SEG = TIPO_CANAL (SEGURO_NOVO/RENOVACAO x DIGITAL/MANUAL) quando ambos existirem", _seg_create,
                  requires_columns=["CANAL", "DS_TIPO_COTACAO"]),
    transform_def("SEG_R02", "Dropar colunas usadas para SEG (CANAL, DS_TIPO_COTACAO)", _seg_drop_inputs),
]


# =========================
# Catálogo para logging (sem fn)
# =========================
transforms_by_phase = {
    "cotacao_join": [
        {k: v for k, v in t.items() if k != "fn"} for t in TRANSFORMS_COTACAO_JOIN
    ],
    "cotacao_seg": [
        {k: v for k, v in t.items() if k != "fn"} for t in TRANSFORMS_COTACAO_SEG
    ],
}

print("✅ CEL 2 carregada: join_spec + lineage + catálogos definidos")
print("• transforms join:", len(TRANSFORMS_COTACAO_JOIN))
print("• transforms seg :", len(TRANSFORMS_COTACAO_SEG))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log MLflow

# COMMAND ----------

# =========================
# Validações
# =========================
ensure_schema(OUT_SCHEMA)

assert_table_exists(SILVER_FACT_FQN)
assert_table_exists(SILVER_RESUMO_FQN)
assert_table_exists(SILVER_DETALHE_FQN)

if table_exists(SEG_TABLE_FQN):
    raise ValueError(f"❌ Seg table já existe: {SEG_TABLE_FQN} (timestamp repetido?)")

# =========================
# MLflow bootstrap
# =========================
_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

child_run_name = f"T_JOIN_{TS_EXEC}"

# prepara catálogo logável (sem fn)
mlflow_transforms_catalog = transforms_catalog_for_logging({
    "cotacao_join": [t if "fn" not in t else {k: v for k, v in t.items() if k != "fn"} for t in TRANSFORMS_COTACAO_JOIN],
    "cotacao_seg":  [t if "fn" not in t else {k: v for k, v in t.items() if k != "fn"} for t in TRANSFORMS_COTACAO_SEG],
})

# logs acumulados
exec_log_all = {"cotacao_join": [], "cotacao_seg": []}

# métricas operacionais
n_gen_in = n_resumo_in = n_detalhe_in = None
n_after_join_1 = n_after_join_2 = None
n_seg_final = None
n_seg_not_null = None

pr_ctx = (
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    if PR_RUN_ID_OVERRIDE
    else mlflow.start_run(run_name=PARENT_RUN_NAME)
)

with pr_ctx as pr:
    if not PR_RUN_ID_OVERRIDE:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "JOIN")
        mlflow.set_tag("run_role", "parent")

    with mlflow.start_run(run_name=child_run_name, nested=True) as cr:
        # -------------------------
        # CR tags
        # -------------------------
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "JOIN")
        mlflow.set_tag("run_role", "child")

        # -------------------------
        # CR params: lineage explícito (inputs/outputs)
        # -------------------------
        mlflow.log_param("silver_fact_fqn", SILVER_FACT_FQN)
        mlflow.log_param("silver_resumo_fqn", SILVER_RESUMO_FQN)
        mlflow.log_param("silver_detalhe_fqn", SILVER_DETALHE_FQN)

        mlflow.log_param("seg_table_fqn", SEG_TABLE_FQN)

        mlflow.log_param("ts_exec", TS_EXEC)
        mlflow.log_param("write_mode", WRITE_MODE)
        mlflow.log_param("enable_rules", str(ENABLE_RULES))
        mlflow.log_param("rule_toggles_present", str(bool(RULE_TOGGLES)))

        # -------------------------
        # Artifacts fixos: lineage + specs + catálogos
        # -------------------------
        mlflow.log_dict(tables_lineage, "tables_lineage.json")
        mlflow.log_dict(join_spec, "join_spec.json")
        mlflow.log_dict(mlflow_transforms_catalog, "transforms_catalog.json")

        # -------------------------
        # Leitura inputs
        # -------------------------
        df_gen = spark.table(SILVER_FACT_FQN)
        df_resumo = spark.table(SILVER_RESUMO_FQN)
        df_detalhe = spark.table(SILVER_DETALHE_FQN)

        cols_gen     = df_gen.columns
        cols_resumo  = df_resumo.columns
        cols_detalhe = df_detalhe.columns

        n_gen_in = df_gen.count()
        n_resumo_in = df_resumo.count()
        n_detalhe_in = df_detalhe.count()

        # -------------------------
        # Monta transforms JOIN_R01/JOIN_R02 com closures (usam df_resumo/df_detalhe)
        # -------------------------
        def JOIN_R01_fn(df: DataFrame) -> DataFrame:
            return _join_1_gen_resumo(df, df_resumo)

        def JOIN_R02_fn(df: DataFrame) -> DataFrame:
            return _join_2_join1_detalhe(df, df_detalhe)

        # injeta fn nos placeholders
        transforms_join_runtime: List[Dict] = []
        for t in TRANSFORMS_COTACAO_JOIN:
            if isinstance(t, dict) and t.get("transform_id") == "JOIN_R01":
                transforms_join_runtime.append(transform_def(
                    "JOIN_R01",
                    t["description"],
                    JOIN_R01_fn,
                    enabled=t.get("enabled", True),
                    requires_columns=t.get("requires_columns", []),
                ))
            elif isinstance(t, dict) and t.get("transform_id") == "JOIN_R02":
                transforms_join_runtime.append(transform_def(
                    "JOIN_R02",
                    t["description"],
                    JOIN_R02_fn,
                    enabled=t.get("enabled", True),
                    requires_columns=t.get("requires_columns", []),
                ))
            else:
                transforms_join_runtime.append(t)

        # -------------------------
        # FASE 0 (interno): capturar n_after_join_1 e n_after_join_2
        # (sem log de regra aqui, apenas observabilidade)
        # -------------------------
        df_after_join_1 = JOIN_R01_fn(df_gen)
        n_after_join_1 = df_after_join_1.count()

        df_after_join_2 = JOIN_R02_fn(df_after_join_1)
        n_after_join_2 = df_after_join_2.count()

        # -------------------------
        # FASE A: cotacao_join (aplica lista completa, grava JOIN_TABLE)
        # -------------------------
        df_join_out, exec_log_join = apply_transforms(
            phase_key="cotacao_join",
            df=df_gen,  # começa do df_gen, e JOIN_R01/JOIN_R02 fazem o resto
            transforms=transforms_join_runtime,
            enable_rules=ENABLE_RULES,
            toggles=RULE_TOGGLES,
        )
        exec_log_all["cotacao_join"] = exec_log_join

        # -------------------------
        # FASE B: cotacao_seg (cria SEG + drops, grava SEG_TABLE)
        # -------------------------
        df_seg_out, exec_log_seg = apply_transforms(
            phase_key="cotacao_seg",
            df=df_join_out,
            transforms=TRANSFORMS_COTACAO_SEG,
            enable_rules=ENABLE_RULES,
            toggles=RULE_TOGGLES,
        )
        exec_log_all["cotacao_seg"] = exec_log_seg

        (df_seg_out.write
            .format("delta")
            .mode(WRITE_MODE)
            .saveAsTable(SEG_TABLE_FQN))

        cols_seg    = df_seg_out.columns
        n_seg_final = df_seg_out.count()
        if SEG_COL in df_seg_out.columns:
            n_seg_not_null = df_seg_out.filter(F.col(SEG_COL).isNotNull()).count()
        else:
            n_seg_not_null = 0

        # profiling input/output
        profiling = {
            "input": {
                "cotacao_generico": {"total_linhas": int(n_gen_in),     "total_colunas": len(cols_gen),     "colunas": list(cols_gen)},
                "corretor_resumo":  {"total_linhas": int(n_resumo_in),  "total_colunas": len(cols_resumo),  "colunas": list(cols_resumo)},
                "corretor_detalhe": {"total_linhas": int(n_detalhe_in), "total_colunas": len(cols_detalhe), "colunas": list(cols_detalhe)},
            },
            "output": {
                "cotacao_seg": {"total_linhas": int(n_seg_final), "total_colunas": len(cols_seg), "colunas": list(cols_seg)},
            },
        }
        mlflow.log_dict(profiling, "profiling/profiling.json")

        # profiling seg (null counts + distinct keys)
        if DO_PROFILE:
            prof_seg = profile_basic(
                df_seg_out,
                name="cotacao_seg",
                key_cols=["CD_NUMERO_COTACAO_AXA", SEG_COL]
            )
            mlflow.log_dict(prof_seg, "profiling_seg.json")

        # -------------------------
        # Artifacts pedidos: contagens SEG e distribuição de status por SEG
        # -------------------------
        seg_counts_art = seg_counts(df_seg_out, SEG_COL)
        mlflow.log_dict(seg_counts_art, "seg_counts.json")

        status_dist_art = status_dist_by_seg(df_seg_out, SEG_COL, STATUS_COL)
        mlflow.log_dict(status_dist_art, "status_dist_by_seg.json")

        # -------------------------
        # Artifacts adicionais: evolução mensal por DS_GRUPO_STATUS (DATA_COTACAO)
        # -------------------------
        log_status_by_month_artifacts(
            df_seg_out,
            date_col="DATA_COTACAO",
            status_col="DS_GRUPO_STATUS",
            artifact_prefix="analysis",
            source_table_fqn=SEG_TABLE_FQN,
        )

        # -------------------------
        # Metrics operacionais
        # -------------------------
        mlflow.log_metric("n_gen_in", int(n_gen_in))
        mlflow.log_metric("n_resumo_in", int(n_resumo_in))
        mlflow.log_metric("n_detalhe_in", int(n_detalhe_in))

        mlflow.log_metric("n_after_join_1", int(n_after_join_1))
        mlflow.log_metric("n_after_join_2", int(n_after_join_2))

        mlflow.log_metric("n_seg_final", int(n_seg_final))
        mlflow.log_metric("n_seg_not_null", int(n_seg_not_null))

        # -------------------------
        # Metrics de transforms
        # -------------------------
        n_total = 0
        n_applied = 0
        n_skipped = 0
        for phase_key, logs in exec_log_all.items():
            for item in logs:
                n_total += 1
                if item["status"] == "APPLIED":
                    n_applied += 1
                else:
                    n_skipped += 1

        mlflow.log_metric("n_transforms_total", int(n_total))
        mlflow.log_metric("n_transforms_applied", int(n_applied))
        mlflow.log_metric("n_transforms_skipped", int(n_skipped))

        # -------------------------
        # Artifacts: execução das transforms
        # -------------------------
        mlflow.log_dict(exec_log_all, "transforms_execution.json")


print("JOIN concluído")
print("• seg  table:", SEG_TABLE_FQN)