# Databricks notebook source
# %pip install pycurl
# %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC #### Imports

# COMMAND ----------

import os
import re
import json
# import pycurl
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
from mlflow.tracking import MlflowClient

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cria schema

# COMMAND ----------

schema_name = "bronze"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configs

# COMMAND ----------

import re
from datetime import datetime
from zoneinfo import ZoneInfo

# =========================
# AJUSTES RECORRENTES
# =========================
PR_RUN_ID_OVERRIDE = "4acddbebaf5242669024d5434bf121fb"
MODE = "COPY_FROM_OLD_BRONZE" # INCREMENTAL_SFTP | REPLAY_SFTP_ALL | COPY_FROM_OLD_BRONZE

# Se MODE == INCREMENTAL_SFTP: esta tabela é usada para cutoff e (opcionalmente) para "já processados"
INCREMENTAL_BASE_TABLE_FQN = "isa_bronze.cotacao_generico"
# Se MODE == COPY_FROM_OLD_BRONZE: de onde copiar a fato
OLD_BRONZE_FACT_TABLE_FQN = "isa_bronze.cotacao_generico"

# =========================
# PARAMS
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"
PARENT_RUN_NAME = "T_PR_INGESTAO"
TARGET_SCHEMA = "bronze"
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
FACT_TABLE_NAME = f"cotacao_generico_{TS_EXEC}"
FACT_TABLE_FQN = f"{TARGET_SCHEMA}.{FACT_TABLE_NAME}"

# =========================
# Fonte SFTP/FTPS (sem logar secrets)
# =========================
HOST = "sftp.axa.com.br"
USER = dbutils.secrets.get(scope="sftp_scope", key="USER")
PASSWORD = dbutils.secrets.get(scope="sftp_scope", key="PASSWORD")

REMOTE_DIR = "/PSW/Arquivos/API/Entrada/GENERICO/COTACAO"
LOCAL_DIR  = "/dbfs/FileStore/tables/"
BRONZE_DIR = "/mnt/isa/bronze/"

FILE_REGEX = re.compile(r"^COTACAO_GENERICO_(?:\d+_)?\d{8}\.csv$")

# =========================
# Leitura CSV (contrato do notebook anexado)
# =========================
CSV_OPTIONS = {
    "header": True,
    "delimiter": "|",
    "quote": "\u0000",
    "escape": "\u0000",
    "encoding": "ISO-8859-1",
}

# =========================
# Tabelas auxiliares (copiar para o schema novo, sem sufixo)
# =========================
SRC_CORRETOR_RESUMO  = "isa_bronze.corretor_resumo"
SRC_CORRETOR_DETALHE = "isa_bronze.corretor_detalhe"

DST_CORRETOR_RESUMO  = f"{TARGET_SCHEMA}.corretor_resumo"
DST_CORRETOR_DETALHE = f"{TARGET_SCHEMA}.corretor_detalhe"

REFRESH_CORRETOR_TABLES = False  # se True, sobrescreve; se False, cria só se não existir

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define funções

# COMMAND ----------

def ensure_schema(schema_name: str) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

def table_exists(fqn: str) -> bool:
    return spark.catalog.tableExists(fqn)

def listar_arquivos_ftp(p_host, p_user, p_password, p_remote_dir):
    buf = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, f"ftps://{p_host}/{p_remote_dir.strip('/')}/")
    c.setopt(c.USERNAME, p_user); c.setopt(c.PASSWORD, p_password)
    c.setopt(c.SSLVERSION, pycurl.SSLVERSION_TLSv1_2)
    c.setopt(c.SSL_VERIFYPEER, 0); c.setopt(c.SSL_VERIFYHOST, 0)
    c.setopt(c.FTP_SSL, pycurl.FTPSSL_CONTROL)
    c.setopt(c.DIRLISTONLY, 1)
    c.setopt(c.WRITEFUNCTION, buf.write)
    c.setopt(c.TIMEOUT, 400)
    try:
        c.perform()
    finally:
        c.close()
    return [s.strip() for s in buf.getvalue().decode("utf-8", "ignore").splitlines() if s.strip()]

def baixar_arquivo_ftp(p_host, p_user, p_password, p_remote_dir, p_nome_arquivo, p_local_dir):
    os.makedirs(p_local_dir, exist_ok=True)
    url = f"ftps://{p_host}/{p_remote_dir.strip('/')}/{p_nome_arquivo}"
    local_path = os.path.join(p_local_dir, p_nome_arquivo)
    print(f"Baixando: {p_nome_arquivo}")
    with open(local_path, "wb") as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.USERNAME, p_user); c.setopt(c.PASSWORD, p_password)
        c.setopt(c.SSLVERSION, pycurl.SSLVERSION_TLSv1_2)
        c.setopt(c.SSL_VERIFYPEER, 0); c.setopt(c.SSL_VERIFYHOST, 0)
        c.setopt(c.FTP_SSL, pycurl.FTPSSL_CONTROL)
        c.setopt(c.CONNECTTIMEOUT, 60); c.setopt(c.TIMEOUT, 1800)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Nao encontrou: {local_path}")
    return local_path

def mover_para_dados_antigos(p_host, p_user, p_password, p_remote_dir, p_nome_arquivo, subdir="DADOS_ANTIGOS"):
    full_dir = p_remote_dir.strip("/")
    origem  = f"/{full_dir}/{p_nome_arquivo}"
    destino = f"/{full_dir}/{subdir}/{p_nome_arquivo}"

    c = pycurl.Curl()
    try:
        c.setopt(c.URL, f"ftps://{p_host}/")
        c.setopt(c.USERNAME, p_user); c.setopt(c.PASSWORD, p_password)
        c.setopt(c.SSLVERSION, pycurl.SSLVERSION_TLSv1_2)
        c.setopt(c.SSL_VERIFYPEER, 0); c.setopt(c.SSL_VERIFYHOST, 0)
        c.setopt(c.FTP_SSL, pycurl.FTPSSL_CONTROL)
        c.setopt(c.CONNECTTIMEOUT, 60); c.setopt(c.TIMEOUT, 400)
        c.setopt(c.QUOTE, [f"RNFR {origem}", f"RNTO {destino}"])
        c.perform()
    finally:
        c.close()

def data_do_nome(nome: str):
    m = re.search(r"(?:^|_)((?:19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01]))\.csv$", nome)
    return m.group(1) if m else None

def get_ultima_data_lote(tabela_fqn: str):
    try:
        df = spark.table(tabela_fqn)
    except:
        return None
    if "TS_ARQ" not in df.columns or df.rdd.isEmpty():
        return None
    return (df.select(F.to_date("TS_ARQ").alias("d"))
              .agg(F.max("d").alias("mx"))
              .collect()[0]["mx"])

def nomes_ja_processados(tabela_fqn: str):
    try:
        df = spark.table(tabela_fqn)
    except:
        return set()
    if "SOURCE_FILE" not in df.columns or df.rdd.isEmpty():
        return set()
    return {r[0] for r in df.select("SOURCE_FILE").distinct().collect()}

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id

def build_cfg_ingestao_dict() -> dict:
    # NÃO incluir USER/PASSWORD aqui
    return {
        "experiment_name": EXPERIMENT_NAME,
        "target_schema": TARGET_SCHEMA,
        "fact_table_fqn": FACT_TABLE_FQN,
        "fact_table_name": FACT_TABLE_NAME,
        "ts_exec": TS_EXEC,
        "mode": MODE,
        "source": {
            "host": HOST,
            "remote_dir": REMOTE_DIR,
            "file_regex": FILE_REGEX.pattern,
        },
        "paths": {
            "local_dir": LOCAL_DIR,
            "bronze_dir": BRONZE_DIR,
        },
        "csv_options": CSV_OPTIONS,
        "incremental_base_table_fqn": INCREMENTAL_BASE_TABLE_FQN if MODE == "INCREMENTAL_SFTP" else None,
        "old_bronze_fact_table_fqn": OLD_BRONZE_FACT_TABLE_FQN if MODE == "COPY_FROM_OLD_BRONZE" else None,
        "corretor_tables": {
            "src_corretor_resumo": SRC_CORRETOR_RESUMO,
            "src_corretor_detalhe": SRC_CORRETOR_DETALHE,
            "dst_corretor_resumo": DST_CORRETOR_RESUMO,
            "dst_corretor_detalhe": DST_CORRETOR_DETALHE,
            "refresh": REFRESH_CORRETOR_TABLES,
        },
    }

def log_ts_arq_profiling(fact_table_fqn: str) -> None:
    """Loga no MLflow um gráfico de barras (linhas por TS_ARQ) e um JSON de profiling."""
    df_fact = spark.table(fact_table_fqn)

    # --- profiling.json ---
    row = df_fact.agg(
        F.min(F.to_date("TS_ARQ")).alias("ts_arq_min"),
        F.max(F.to_date("TS_ARQ")).alias("ts_arq_max"),
        F.count("*").alias("total_linhas"),
    ).collect()[0]
    colunas = df_fact.columns
    profiling = {
        "total_linhas": int(row["total_linhas"]),
        "total_colunas": len(colunas),
        "colunas": colunas,
        "ts_arq_min": str(row["ts_arq_min"]),
        "ts_arq_max": str(row["ts_arq_max"]),
    }
    mlflow.log_dict(profiling, "profiling/profiling.json")

    # --- contagem por TS_ARQ ---
    df_counts = (
        df_fact
        .groupBy(F.to_date("TS_ARQ").alias("data"))
        .count()
        .orderBy("data")
        .toPandas()
    )
    df_counts["data"] = df_counts["data"].astype(str)

    fig, ax = plt.subplots(figsize=(max(8, len(df_counts) * 0.5), 5))
    ax.bar(df_counts["data"], df_counts["count"])
    ax.set_xlabel("TS_ARQ")
    ax.set_ylabel("Linhas")
    ax.set_title("Contagem de linhas por TS_ARQ")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    mlflow.log_figure(fig, "profiling/ts_arq_contagem.png")
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verifica mode e params

# COMMAND ----------

assert MODE in {"INCREMENTAL_SFTP", "REPLAY_SFTP_ALL", "COPY_FROM_OLD_BRONZE"}, f"MODE inválido: {MODE}"

ensure_schema(TARGET_SCHEMA)

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

if table_exists(FACT_TABLE_FQN):
    raise ValueError(f"A tabela destino já existe: {FACT_TABLE_FQN} (timestamp repetido?)")

print("Bootstrap ok")
print("• experiment:", EXPERIMENT_NAME)
print("• fact_table:", FACT_TABLE_FQN)
print("• mode:", MODE)
print("• pr_run_id_override:", PR_RUN_ID_OVERRIDE or "(novo container)")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Executa ingestões

# COMMAND ----------

def copy_corretor_tables():
    if REFRESH_CORRETOR_TABLES or (not table_exists(DST_CORRETOR_RESUMO)):
        (spark.table(SRC_CORRETOR_RESUMO)
              .write.format("delta")
              .mode("overwrite")
              .saveAsTable(DST_CORRETOR_RESUMO))
    if REFRESH_CORRETOR_TABLES or (not table_exists(DST_CORRETOR_DETALHE)):
        (spark.table(SRC_CORRETOR_DETALHE)
              .write.format("delta")
              .mode("overwrite")
              .saveAsTable(DST_CORRETOR_DETALHE))

def run_ingestao_into_new_fact_table():
    processados_no_run = []
    linhas_por_arquivo = []
    arquivos_listados = []
    candidatos = []

    if MODE == "COPY_FROM_OLD_BRONZE":
        df_final = spark.table(OLD_BRONZE_FACT_TABLE_FQN)
        n_total = df_final.count()

        (df_final.write
                .format("delta")
                .mode("overwrite")
                .saveAsTable(FACT_TABLE_FQN))

        return {
            "arquivos_listados": arquivos_listados,
            "candidatos": candidatos,
            "processados_no_run": processados_no_run,
            "linhas_por_arquivo": linhas_por_arquivo,
            "n_total": n_total,
        }

    # Modos SFTP/FTPS
    arquivos_listados = listar_arquivos_ftp(HOST, USER, PASSWORD, REMOTE_DIR)
    arquivos = [f for f in arquivos_listados if re.fullmatch(FILE_REGEX, f)]

    if MODE == "INCREMENTAL_SFTP":
        ultima_data = get_ultima_data_lote(INCREMENTAL_BASE_TABLE_FQN)
        cutoff = ultima_data.strftime("%Y%m%d") if ultima_data else None
        ja = nomes_ja_processados(INCREMENTAL_BASE_TABLE_FQN)
    else:
        cutoff = None
        ja = set()

    cand_all = [(f, data_do_nome(f)) for f in arquivos if data_do_nome(f)]
    cand_all.sort(key=lambda x: (x[1], x[0]))

    if MODE == "INCREMENTAL_SFTP":
        candidatos = [(f, d) for (f, d) in cand_all if (f not in ja) and (cutoff is None or d >= cutoff)]
    else:
        candidatos = cand_all

    if not candidatos:
        return {
            "arquivos_listados": arquivos_listados,
            "candidatos": candidatos,
            "processados_no_run": processados_no_run,
            "linhas_por_arquivo": linhas_por_arquivo,
            "n_total": 0,
        }

    total = 0
    wrote_any = False

    for nome, d_yyyymmdd in candidatos:
        caminho_local = baixar_arquivo_ftp(HOST, USER, PASSWORD, REMOTE_DIR, nome, LOCAL_DIR)

        bronze_file = f"{BRONZE_DIR.rstrip('/')}/{nome}"
        dbutils.fs.mv(f"file:{caminho_local}", bronze_file, True)

        reader = spark.read
        for k, v in CSV_OPTIONS.items():
            reader = reader.option(k, v)
        df = reader.csv(bronze_file)

        df = df.toDF(*[c.replace('"', '').replace("'", "") for c in df.columns])
        for c, t in df.dtypes:
            if t == "string":
                df = df.withColumn(c, F.regexp_replace(F.col(c), '"', ''))

        df = (df
              .withColumn("TS_ARQ", F.to_timestamp(F.to_date(F.lit(d_yyyymmdd), "yyyyMMdd")))
              .withColumn("TS_ATUALIZACAO", F.current_timestamp())
              .withColumn("SOURCE_FILE", F.lit(nome))
        )

        mode = "overwrite" if not wrote_any else "append"
        (df.write
           .format("delta")
           .mode(mode)
           .saveAsTable(FACT_TABLE_FQN))
        wrote_any = True

        n = df.count()
        total += n
        processados_no_run.append(nome)
        linhas_por_arquivo.append({"file": nome, "yyyymmdd": d_yyyymmdd, "n_rows": n})
        print(f"{nome} ({d_yyyymmdd}) -> +{n} linhas")

    return {
        "arquivos_listados": arquivos_listados,
        "candidatos": candidatos,
        "processados_no_run": processados_no_run,
        "linhas_por_arquivo": linhas_por_arquivo,
        "n_total": total,
    }


# -------------------------
# Runs (PR + CR)
# -------------------------
pr_ctx = (
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    if PR_RUN_ID_OVERRIDE
    else mlflow.start_run(run_name=PARENT_RUN_NAME)
)

with pr_ctx as pr:
    if not PR_RUN_ID_OVERRIDE:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "INGESTAO")
        mlflow.set_tag("run_role", "parent")

    with mlflow.start_run(run_name=f"T_INGESTAO_{TS_EXEC}", nested=True) as cr:
        mlflow.set_tag("pipeline_tipo", "T")
        mlflow.set_tag("etapa", "INGESTAO")
        mlflow.set_tag("run_role", "child")
        mlflow.set_tag("ingestao_modo", MODE)

        mlflow.log_param("target_schema", TARGET_SCHEMA)
        mlflow.log_param("fact_table_fqn", FACT_TABLE_FQN)
        mlflow.log_param("fact_table_name", FACT_TABLE_NAME)
        mlflow.log_param("file_regex", FILE_REGEX.pattern)
        mlflow.log_param("remote_dir", REMOTE_DIR)
        mlflow.log_param("delimiter", CSV_OPTIONS["delimiter"])
        mlflow.log_param("encoding", CSV_OPTIONS["encoding"])
        mlflow.log_param("refresh_corretor_tables", str(REFRESH_CORRETOR_TABLES))

        if MODE == "INCREMENTAL_SFTP":
            mlflow.log_param("incremental_base_table_fqn", INCREMENTAL_BASE_TABLE_FQN)
            mlflow.log_param("cutoff_enabled", "true")
        else:
            mlflow.log_param("cutoff_enabled", "false")

        mlflow.log_dict(build_cfg_ingestao_dict(), "cfg_ingestao.json")

        # ---- executa ingestão ----
        out = run_ingestao_into_new_fact_table()

        # métricas base
        mlflow.log_metric("n_arquivos_listados", len(out["arquivos_listados"]))
        mlflow.log_metric("n_candidatos", len(out["candidatos"]))
        mlflow.log_metric("n_processados", len(out["processados_no_run"]))
        mlflow.log_metric("n_linhas_total", int(out["n_total"]))

        # artefatos base
        mlflow.log_text("\n".join(out["arquivos_listados"]), "arquivos_ftp.txt")
        mlflow.log_dict([{"file": f, "yyyymmdd": d} for (f, d) in out["candidatos"]], "candidatos.json")
        mlflow.log_dict(out["processados_no_run"], "processados.json")
        mlflow.log_dict(out["linhas_por_arquivo"], "linhas_por_arquivo.json")

        # profiling TS_ARQ
        if out["n_total"] > 0:
            log_ts_arq_profiling(FACT_TABLE_FQN)

        # pós-processo SFTP: mover para DADOS_ANTIGOS
        n_move_ok, n_move_fail = 0, 0
        if MODE in {"INCREMENTAL_SFTP", "REPLAY_SFTP_ALL"} and out["processados_no_run"]:
            for nome in out["processados_no_run"]:
                try:
                    mover_para_dados_antigos(HOST, USER, PASSWORD, REMOTE_DIR, nome, "DADOS_ANTIGOS")
                    n_move_ok += 1
                except Exception as e:
                    n_move_fail += 1
                    print(f"Erro ao mover {nome}: {e}")

        mlflow.log_metric("n_move_ok", n_move_ok)
        mlflow.log_metric("n_move_fail", n_move_fail)

        copy_corretor_tables()

print("Concluido")
print("• fato:", FACT_TABLE_FQN)
print("• corretor_resumo:", DST_CORRETOR_RESUMO)
print("• corretor_detalhe:", DST_CORRETOR_DETALHE)