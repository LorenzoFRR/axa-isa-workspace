# Databricks notebook source
from datetime import datetime
from zoneinfo import ZoneInfo
import mlflow

# =========================
# MLflow / Estrutura
# =========================
EXPERIMENT_NAME = "/Workspace/Users/psw.service@pswdigital.com.br/TESTE_ML_NOVO/TESTE/ISA_EXP"

PR_COMP_NAME = "T_PR_COMP"   # container (sem logging)
COMP_VERSAO = "V8"
VERSAO_REF = COMP_VERSAO
TS_EXEC = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

RUN_COMP_EXEC = run_name_vts("T_COMP")

# =========================
# Override: preencha para reutilizar container PR existente
# =========================
PR_COMP_RUN_ID_OVERRIDE = '9db9936565ce4512aab8966c638c6897'

# =========================
# INPUT
# =========================
SCORED_TABLE_FQN = "gold.cotacao_inferencia_mode_b_20260309_122422"  # <<< AJUSTE
FILTER_BY_SEG = True
SEG_TARGET = "SEGURO_NOVO_MANUAL"  # <<< AJUSTE

# =========================
# Colunas esperadas
# =========================
STATUS_COL = "DS_GRUPO_STATUS"
P_COL = "p_emitida"
PRED_COL = "pred_emitida"
DATE_COL = "DATA_COTACAO"

SEG_COL = "SEG"  # só se FILTER_BY_SEG=True

print("✅ CONFIG carregada")
print("• input:", SCORED_TABLE_FQN)
print("• run:", RUN_COMP_EXEC)

# COMMAND ----------

import os
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id

def log_png(fig, artifact_path: str, dpi: int = 150):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

def status_simple_col():
    # padroniza para 2 classes principais (para cores)
    s = F.upper(F.trim(F.col(STATUS_COL).cast("string")))
    return (
        F.when(s == F.lit("EMITIDA"), F.lit("Emitida"))
         .when(s == F.lit("PERDIDA"), F.lit("Perdida"))
         .otherwise(F.col(STATUS_COL).cast("string"))
    )

def ensure_required_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Colunas faltando no input: {missing}")

# Paleta única (pastel) para todos os gráficos (exceto topk)
PALETTE_STATUS = {
    "Emitida":  "#A8D5BA",  # verde pastel
    "Perdida":  "#F4B6B6",  # vermelho/rosa pastel
    "_default": "#CFCFCF",  # cinza pastel p/ outros status
}

print("✅ Helpers carregados")

# COMMAND ----------

# Experiment
_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# encerra run pendurada no runtime
while mlflow.active_run() is not None:
    mlflow.end_run()

# PR container (sem logs)
if PR_COMP_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_COMP_RUN_ID_OVERRIDE)
    PR_COMP_RUN_ID = PR_COMP_RUN_ID_OVERRIDE
    _pr_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=PR_COMP_NAME)
    PR_COMP_RUN_ID = mlflow.active_run().info.run_id
    _pr_status = "nova"

# Exec run versionada (todos os logs vão acontecer aqui)
mlflow.start_run(run_name=RUN_COMP_EXEC, nested=True)

# -------------------------
# LOGGING (somente exec run)
# -------------------------
mlflow.set_tag("stage", "COMP")
mlflow.set_tag("run_role", "exec")
mlflow.set_tag("seg_target", SEG_TARGET)
mlflow.set_tag("versao_ref", VERSAO_REF)

mlflow.log_param("ts_exec", TS_EXEC)
mlflow.log_param("comp_versao", COMP_VERSAO)
mlflow.log_param("versao_ref", VERSAO_REF)
mlflow.log_param("run_suffix", RUN_SUFFIX)
mlflow.log_param("seg_target", SEG_TARGET)
mlflow.log_param("filter_by_seg", FILTER_BY_SEG)

mlflow.log_param("pr_run_id", PR_COMP_RUN_ID)
mlflow.log_param("scored_table_fqn", SCORED_TABLE_FQN)

print("✅ MLflow runs abertas")
print("• PR:", PR_COMP_NAME, f"({_pr_status})")
print("• RUN:", RUN_COMP_EXEC)

# COMMAND ----------

# -------------------------
# Load + filtros
# -------------------------
df0 = spark.table(SCORED_TABLE_FQN)

ensure_required_cols(df0, [STATUS_COL, P_COL, PRED_COL, DATE_COL])
if FILTER_BY_SEG:
    ensure_required_cols(df0, [SEG_COL])
    df0 = df0.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))

df = (
    df0
    .withColumn("status_simple", status_simple_col())
    .withColumn("p_emitida_d", F.col(P_COL).cast("double"))
    .withColumn("pred_emitida_int", F.col(PRED_COL).cast("int"))
    .withColumn("label_real", F.when(F.col("status_simple") == F.lit("Emitida"), F.lit(1)).otherwise(F.lit(0)))
    .withColumn("DATA_COTACAO_dt", F.to_date(F.col(DATE_COL)))
    .filter(
        F.col("p_emitida_d").isNotNull() &
        F.col("pred_emitida_int").isNotNull() &
        F.col("label_real").isNotNull() &
        F.col("DATA_COTACAO_dt").isNotNull()
    )
    .withColumn("MES", F.date_format(F.col("DATA_COTACAO_dt"), "yyyy-MM"))
    .cache()
)

N = df.count()
if N == 0:
    raise ValueError("❌ Após filtros e casts, não há linhas para comparação.")

# -------------------------
# Contagens reais e preditas
# -------------------------
pdf_real = (df.groupBy("label_real").count().orderBy("label_real").toPandas())
pdf_pred = (df.groupBy("pred_emitida_int").count().orderBy("pred_emitida_int").toPandas())

count_labels_reais = {int(r["label_real"]): int(r["count"]) for _, r in pdf_real.iterrows()}
count_pred_emitida = {int(r["pred_emitida_int"]): int(r["count"]) for _, r in pdf_pred.iterrows()}

n_pos = count_labels_reais.get(1, 0)
base_rate = (n_pos / N) if N else None

# -------------------------
# Confusion matrix + recall
# -------------------------
pdf_cm = (df.groupBy("label_real", "pred_emitida_int").count().toPandas())

def _cm(y, yhat):
    r = pdf_cm[(pdf_cm["label_real"] == y) & (pdf_cm["pred_emitida_int"] == yhat)]
    return int(r["count"].iloc[0]) if len(r) else 0

tn = _cm(0, 0)
fp = _cm(0, 1)
fn = _cm(1, 0)
tp = _cm(1, 1)

recall = (tp / (tp + fn)) if (tp + fn) > 0 else None

# -------------------------
# Average Precision (AP) — igual ao notebook de referência (sem coletar tudo)
# AP = média das precisões nos ranks onde label_real=1
# -------------------------
ap = None
if n_pos > 0:
    w_desc = Window.orderBy(F.col("p_emitida_d").desc())
    w_cum = w_desc.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    df_ap = (
        df.select("label_real", "p_emitida_d")
          .withColumn("rn", F.row_number().over(w_desc))
          .withColumn("tp_cum", F.sum(F.col("label_real")).over(w_cum))
          .withColumn("precision_at_i", F.col("tp_cum") / F.col("rn"))
    )
    ap = df_ap.filter(F.col("label_real") == 1).agg(F.avg("precision_at_i").alias("ap")).collect()[0]["ap"]
    ap = float(ap) if ap is not None else None

# -------------------------
# AUC PR (areaUnderPR) via BinaryClassificationMetrics (RDD)
# -------------------------
# Observação: esperado tuple(score, label) como float
rdd = df.select(F.col("p_emitida_d").cast("double"), F.col("label_real").cast("double")).rdd.map(lambda r: (float(r[0]), float(r[1])))
auc_pr = float(BinaryClassificationMetrics(rdd).areaUnderPR)

# -------------------------
# JSON final (único log)
# -------------------------
metricas_json = {
    "average_precision": ap,
    "auc_pr": auc_pr,
    "base_rate": float(base_rate) if base_rate is not None else None,
    "recall": float(recall) if recall is not None else None,
    "count_labels_reais": count_labels_reais,
    "count_pred_emitida": count_pred_emitida,
    "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    "n_total": int(N),
}

mlflow.log_dict(metricas_json, "metricas_json.json")

print("✅ metricas_json.json logado")
print(metricas_json)

# COMMAND ----------

# ==========================================================
# Top-K% (1..100) — exatamente como referência
# ==========================================================
df_rank = (
    df.select("label_real", "p_emitida_d")
      .withColumn("label_real", F.col("label_real").cast("int"))
      .withColumn("p_emitida_d", F.col("p_emitida_d").cast("double"))
)

N_rank = df_rank.count()
n_pos_total = int(df_rank.agg(F.sum("label_real").alias("n_pos")).collect()[0]["n_pos"] or 0)
base_rate_local = (n_pos_total / N_rank) if N_rank else 0.0

w = Window.orderBy(F.col("p_emitida_d").desc())
df_ord = (
    df_rank
    .withColumn("rn", F.row_number().over(w))
    .withColumn("pos_cum", F.sum("label_real").over(w.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
    .cache()
)
_ = df_ord.count()

K_PCTS = [k/100.0 for k in range(1, 101)]
rows_k = []
for k in K_PCTS:
    k_n = int(round(N_rank * k))
    k_n = max(k_n, 1)
    k_n = min(k_n, N_rank)
    rows_k.append((float(k), int(k_n)))

df_k = spark.createDataFrame(rows_k, ["k_pct", "k_n"])

df_k_metrics = (
    df_k
    .join(df_ord.select("rn", "pos_cum"), df_ord["rn"] == df_k["k_n"], "left")
    .withColumnRenamed("pos_cum", "pos_in_k")
    .drop("rn")
    .withColumn("precision_at_k", F.col("pos_in_k") / F.col("k_n"))
    .withColumn("recall_at_k", F.when(F.lit(n_pos_total) > 0, F.col("pos_in_k") / F.lit(n_pos_total)))
    .withColumn("lift_at_k", F.when(F.lit(base_rate_local) > 0, F.col("precision_at_k") / F.lit(base_rate_local)))
    .withColumn("precision_random", F.lit(base_rate_local))
    .withColumn("recall_random", F.col("k_pct"))
    .withColumn("lift_random", F.lit(1.0))
    .orderBy("k_pct")
)

pdf_k = df_k_metrics.toPandas()
pdf_k["k_pct_100"] = pdf_k["k_pct"] * 100.0

# ---- plot único com subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(pdf_k["k_pct_100"], pdf_k["precision_at_k"], marker="o", markersize=2, linewidth=1, label="precision@k (modelo)")
axs[0].plot(pdf_k["k_pct_100"], pdf_k["precision_random"], linestyle="--", linewidth=1, label="precision@k (aleatório)")
axs[0].set_ylabel("Precision@K")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(pdf_k["k_pct_100"], pdf_k["recall_at_k"], marker="o", markersize=2, linewidth=1, label="recall@k (modelo)")
axs[1].plot(pdf_k["k_pct_100"], pdf_k["recall_random"], linestyle="--", linewidth=1, label="recall@k (aleatório)")
axs[1].set_ylabel("Recall@K")
axs[1].grid(True)
axs[1].legend()

axs[2].plot(pdf_k["k_pct_100"], pdf_k["lift_at_k"], marker="o", markersize=2, linewidth=1, label="lift@k (modelo)")
axs[2].plot(pdf_k["k_pct_100"], pdf_k["lift_random"], linestyle="--", linewidth=1, label="lift@k (aleatório)")
axs[2].set_xlabel("Top-K (%)")
axs[2].set_ylabel("Lift@K")
axs[2].grid(True)
axs[2].legend()

fig.suptitle("Top-K%: Precision / Recall / Lift (modelo vs baseline aleatório)", y=1.02)
plt.tight_layout()

log_png(fig, "metricas_visuais/topk_precision_recall_lift.png")
print("✅ metricas_visuais/topk_precision_recall_lift.png logado")

# COMMAND ----------

# ==========================================================
# Evolução mensal
# ==========================================================
# Contagens por MES x status_simple
df_ms = (df.groupBy("MES", "status_simple").count().orderBy("MES"))
pdf_ms = df_ms.toPandas()
if pdf_ms.empty:
    raise ValueError("❌ Sem dados para evolução mensal.")

# Pivot contagem
pdf_cnt = pdf_ms.pivot_table(index="MES", columns="status_simple", values="count", fill_value=0).reset_index()

# Percentual
pdf_pct = pdf_cnt.copy()
status_cols = [c for c in pdf_cnt.columns if c != "MES"]
row_sum = pdf_pct[status_cols].sum(axis=1).replace(0, np.nan)
for c in status_cols:
    pdf_pct[c] = (pdf_pct[c] / row_sum) * 100.0
pdf_pct = pdf_pct.fillna(0.0)

# Linhas de média p_emitida por mês:
# - geral
# - Emitida (status_simple == Emitida)
# - Perdida (status_simple == Perdida)
df_mean_all = df.groupBy("MES").agg(F.avg("p_emitida_d").alias("p_mean_all")).orderBy("MES")
df_mean_by_status = (df.groupBy("MES", "status_simple")
                       .agg(F.avg("p_emitida_d").alias("p_mean"))
                       .orderBy("MES", "status_simple"))

pdf_mean_all = df_mean_all.toPandas()
pdf_mean_bs = df_mean_by_status.toPandas()

mes = pdf_cnt["MES"].tolist()
x = np.arange(len(mes))

# alinhamento das linhas
pdf_mean_all = pdf_mean_all.set_index("MES").reindex(mes).reset_index()
def _mean_series(status_name: str):
    sub = pdf_mean_bs[pdf_mean_bs["status_simple"] == status_name].copy()
    sub = sub.set_index("MES").reindex(mes).reset_index()
    return sub["p_mean"].astype(float).values

y_all = pdf_mean_all["p_mean_all"].astype(float).values
y_emitida = _mean_series("Emitida") if "Emitida" in pdf_mean_bs["status_simple"].unique() else np.full(len(mes), np.nan)
y_perdida = _mean_series("Perdida") if "Perdida" in pdf_mean_bs["status_simple"].unique() else np.full(len(mes), np.nan)

# Cores exigidas
COLORS = {
    "Emitida": "green",
    "Perdida": "red",
}

# ---- plot (2 subplots) com paleta pastel nas barras e linhas pretas com estilos diferentes
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Paleta pastel (barras)
PASTEL = PALETTE_STATUS

# (A) Contagens stacked
bottom = np.zeros(len(mes))
for st in status_cols:
    axs[0].bar(
        x,
        pdf_cnt[st].values,
        bottom=bottom,
        label=st,
        color=PASTEL.get(st, PASTEL["_default"])
    )
    bottom += pdf_cnt[st].values

axs[0].set_ylabel("Qtd de cotações")
axs[0].set_title("Evolução mensal — contagem por DS_GRUPO_STATUS")
axs[0].grid(True, axis="y")
axs[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1))

# (B) Percentual stacked + linhas (todas pretas, estilos diferentes)
bottom = np.zeros(len(mes))
for st in status_cols:
    axs[1].bar(
        x,
        pdf_pct[st].values,
        bottom=bottom,
        label=st,
        color=PASTEL.get(st, PASTEL["_default"])
    )
    bottom += pdf_pct[st].values

axs[1].set_ylabel("% por status (100%)")
axs[1].set_title("Evolução mensal — distribuição percentual + média p_emitida")
axs[1].grid(True, axis="y")

# eixo secundário para p_emitida
ax2 = axs[1].twinx()

# linhas pretas com estilos distintos
ax2.plot(x, y_all, color="black", linestyle="-",  marker="o", linewidth=1.5, label="p_mean (Geral)")
ax2.plot(x, y_emitida, color="black", linestyle="--", marker="x", linewidth=1.5, label="p_mean (Emitida)")
ax2.plot(x, y_perdida, color="black", linestyle="-.", marker="s", linewidth=1.5, label="p_mean (Perdida)")

ax2.set_ylabel("p_emitida (média)")

# legenda combinada
h1, l1 = axs[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.02, 1))

axs[1].set_xticks(x)
axs[1].set_xticklabels(mes, rotation=45, ha="right")
axs[1].set_xlabel("Mês (YYYY-MM)")

plt.tight_layout()
log_png(fig, "metricas_visuais/evolucao_mensal.png")
print("✅ metricas_visuais/evolucao_mensal.png (pastel bars + black styled lines) logado")

# COMMAND ----------

# ==========================================================
# Densidade (hist density) + Histograma (contagem bruta) de p_emitida
# por DS_GRUPO_STATUS (2 subplots) — mesma paleta global
# ==========================================================
df_s = (
    df.select("status_simple", "p_emitida_d")
      .filter(F.col("p_emitida_d").isNotNull())
      .sample(withReplacement=False, fraction=1.0, seed=42)
      .limit(200_000)
)

pdf_s = df_s.toPandas()
if pdf_s.empty:
    raise ValueError("❌ Sem dados para plot densidade/hist de p_emitida.")

# manter apenas Emitida/Perdida (padrão do comparativo)
pdf_s = pdf_s[pdf_s["status_simple"].isin(["Emitida", "Perdida"])].copy()
if pdf_s.empty:
    raise ValueError("❌ Sem linhas de Emitida/Perdida para o plot densidade/hist.")

bins = np.linspace(0, 1, 51)  # 50 bins
centers = (bins[:-1] + bins[1:]) / 2.0
bin_width = bins[1] - bins[0]

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# (A) Densidade em formato de histograma (density=True)
for st in ["Emitida", "Perdida"]:
    x = pdf_s.loc[pdf_s["status_simple"] == st, "p_emitida_d"].astype(float).values
    if len(x) == 0:
        continue
    hist_density, _ = np.histogram(x, bins=bins, density=True)
    axes[0].bar(
        centers,
        hist_density,
        width=bin_width,
        alpha=0.45,
        label=st,
        color=PALETTE_STATUS.get(st, PALETTE_STATUS["_default"]),
        edgecolor="white",
        linewidth=0.5
    )

axes[0].set_title("p_emitida — histograma em densidade por DS_GRUPO_STATUS")
axes[0].set_ylabel("densidade")
axes[0].grid(True, axis="y")
axes[0].legend()

# (B) Histograma de contagem bruta (density=False)
for st in ["Emitida", "Perdida"]:
    x = pdf_s.loc[pdf_s["status_simple"] == st, "p_emitida_d"].astype(float).values
    if len(x) == 0:
        continue
    hist_count, _ = np.histogram(x, bins=bins, density=False)
    axes[1].bar(
        centers,
        hist_count,
        width=bin_width,
        alpha=0.45,
        label=st,
        color=PALETTE_STATUS.get(st, PALETTE_STATUS["_default"]),
        edgecolor="white",
        linewidth=0.5
    )

axes[1].set_title("p_emitida — histograma de contagem bruta por DS_GRUPO_STATUS")
axes[1].set_xlabel("p_emitida")
axes[1].set_ylabel("contagem")
axes[1].grid(True, axis="y")
axes[1].legend()

plt.tight_layout()
log_png(fig, "metricas_visuais/densidade_hist_p_emitida.png")

print("✅ metricas_visuais/densidade_hist_p_emitida.png (densidade + contagem bruta) logado")

# COMMAND ----------

import os, tempfile
import pandas as pd

def log_csv(df_csv: pd.DataFrame, artifact_path: str):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        df_csv.to_csv(fp, index=False)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

# ==========================================================
# 1) Top-K% data (reusa pdf_k da célula do top-k)
# Espera existir: pdf_k com colunas:
#  - k_pct, k_pct_100, k_n, pos_in_k, precision_at_k, recall_at_k, lift_at_k
#  - precision_random, recall_random, lift_random
# ==========================================================
topk_long = []
for _, r in pdf_k.iterrows():
    for serie, val in [
        ("precision_model", r["precision_at_k"]),
        ("precision_random", r["precision_random"]),
        ("recall_model", r["recall_at_k"]),
        ("recall_random", r["recall_random"]),
        ("lift_model", r["lift_at_k"]),
        ("lift_random", r["lift_random"]),
    ]:
        topk_long.append({
            "plot": "topk",
            "x_type": "k_pct",
            "x": float(r["k_pct"]),         # 0..1
            "x_label": float(r["k_pct_100"]),# 1..100
            "serie": serie,
            "y": float(val) if val is not None else None,
            "k_n": int(r["k_n"]),
            "pos_in_k": int(r.get("pos_in_k", 0)) if pd.notna(r.get("pos_in_k", None)) else None,
        })
df_topk_long = pd.DataFrame(topk_long)

# ==========================================================
# 2) Evolução mensal data (reusa pdf_cnt, pdf_pct, y_all/y_emitida/y_perdida)
# Espera existir:
#  - pdf_cnt: colunas MES + status_cols (contagens)
#  - pdf_pct: colunas MES + status_cols (percentuais 0..100)
#  - mes (lista), x (índices)
#  - y_all, y_emitida, y_perdida (arrays alinhados a 'mes')
# ==========================================================
monthly_long = []

# stacked contagem
for _, r in pdf_cnt.iterrows():
    for st in status_cols:
        monthly_long.append({
            "plot": "monthly_count",
            "x_type": "MES",
            "x": r["MES"],
            "x_label": r["MES"],
            "serie": f"count_{st}",
            "y": float(r[st]),
        })

# stacked percentual
for _, r in pdf_pct.iterrows():
    for st in status_cols:
        monthly_long.append({
            "plot": "monthly_pct",
            "x_type": "MES",
            "x": r["MES"],
            "x_label": r["MES"],
            "serie": f"pct_{st}",
            "y": float(r[st]),
        })

# linhas p_emitida médias (todas pretas no plot; aqui só dados)
for i, m in enumerate(mes):
    monthly_long.append({"plot": "monthly_pmean", "x_type": "MES", "x": m, "x_label": m, "serie": "p_mean_all", "y": float(y_all[i]) if pd.notna(y_all[i]) else None})
    monthly_long.append({"plot": "monthly_pmean", "x_type": "MES", "x": m, "x_label": m, "serie": "p_mean_emitida", "y": float(y_emitida[i]) if pd.notna(y_emitida[i]) else None})
    monthly_long.append({"plot": "monthly_pmean", "x_type": "MES", "x": m, "x_label": m, "serie": "p_mean_perdida", "y": float(y_perdida[i]) if pd.notna(y_perdida[i]) else None})

df_monthly_long = pd.DataFrame(monthly_long)

# ==========================================================
# 3) Densidade/hist data (NOVO):
#  - topo: density histogram (density=True)
#  - baixo: count histogram (density=False)
# Espera existir:
#  - pdf_s com colunas status_simple e p_emitida_d
#  - bins, centers (gerados na célula do plot densidade/hist)
# ==========================================================
dens_long = []
count_long = []

edges = bins  # só para clareza

for st in ["Emitida", "Perdida"]:
    xs = pdf_s.loc[pdf_s["status_simple"] == st, "p_emitida_d"].astype(float).values
    if len(xs) == 0:
        continue

    hist_density, _ = np.histogram(xs, bins=edges, density=True)
    hist_count, _ = np.histogram(xs, bins=edges, density=False)

    for i in range(len(centers)):
        dens_long.append({
            "plot": "p_density_hist",
            "x_type": "p_emitida_bin_center",
            "x": float(centers[i]),
            "x_label": float(centers[i]),
            "serie": f"density_{st}",
            "y": float(hist_density[i]),
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i+1]),
        })
        count_long.append({
            "plot": "p_hist_count",
            "x_type": "p_emitida_bin_center",
            "x": float(centers[i]),
            "x_label": float(centers[i]),
            "serie": f"count_{st}",
            "y": float(hist_count[i]),
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i+1]),
        })

df_dens_long = pd.DataFrame(dens_long)
df_count_long = pd.DataFrame(count_long)

# ==========================================================
# Consolida CSV único
# ==========================================================
df_plots = pd.concat([df_topk_long, df_monthly_long, df_dens_long, df_count_long], ignore_index=True)

log_csv(df_plots, "metricas_visuais/dados_plots.csv")
print("✅ CSV logado:", "metricas_visuais/dados_plots.csv")
print("Linhas:", len(df_plots), "| Colunas:", df_plots.columns.tolist())

# COMMAND ----------

