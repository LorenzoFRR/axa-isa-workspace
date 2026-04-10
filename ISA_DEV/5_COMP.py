# Databricks notebook source
# MAGIC %md
# MAGIC ## Configs

# COMMAND ----------

from datetime import datetime
from zoneinfo import ZoneInfo
import mlflow

# =========================
# MLflow / Estrutura
# =========================
EXPERIMENT_NAME = "/Users/psw.service@pswdigital.com.br/ISA_DEV/ISA_DEV"
PR_COMP_NAME            = "T_PR_COMP"
MODE_CODE               = "D"
COMP_VERSAO             = "V11.0.0"
VERSAO_REF              = COMP_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar container PR existente
PR_COMP_RUN_ID_OVERRIDE = "9db9936565ce4512aab8966c638c6897"

# =========================
# INPUT
# =========================
INFERENCIA_TABLE_FQN = "gold.cotacao_inferencia_mode_d_seguro_novo_manual_20260406_150924"  # <<< AJUSTE
SEG_TARGET           = "SEGURO_NOVO_MANUAL"                   # <<< AJUSTE

# =========================
# MODELOS
# =========================
# Se vazio ([]), infere automaticamente das colunas p_emitida_* da tabela
MODEL_IDS = []

# =========================
# REFERÊNCIA AO TREINO
# =========================
# run_id do exec run T_TREINO — necessário para carregar os thresholds de cada modelo.
# Se vazio, thresholds não são anotados nos gráficos.
TREINO_EXEC_RUN_ID = "298ed3dae51b4bfba9cf7a6408d3f400"   # <<< AJUSTE

# =========================
# REFERÊNCIA AO JOIN
# =========================
# run_id do exec run do JOIN (run_role=exec, etapa=JOIN) — necessário para barras de status no gráfico temporal.
# Se vazio, barras não são adicionadas ao gráfico.
JOIN_EXEC_RUN_ID = "28fbd7c3809e4eb0b437956b7af15dd8"   # <<< AJUSTE

# =========================
# REFERÊNCIA AO CLUSTERING (apenas MODE_D)
# =========================
# run_id do exec run T_CLUSTERING_FIT — opcional.
# Se preenchido, enriquece clustering/cluster_profile.json com centroides do fitting.
CLF_FIT_EXEC_RUN_ID = "c19f81e6cd0448f4af52e0713ddf05a9"  # <<< AJUSTE (apenas MODE_D)

# =========================
# PARÂMETROS DE ANÁLISE
# =========================
TOPK_STEP    = 1    # passo da varredura K% (1 = K=1%,2%,...,100%)
TOPK_REF_PCT = 15   # K% de referência para Bloco 5 (temporal) e Bloco 6 (seleção)

# =========================
# Colunas esperadas
# =========================
STATUS_COL       = "DS_GRUPO_STATUS"
DATE_COL         = "DATA_COTACAO"
SEG_COL          = "SEG"
CLF_CORRETOR_COL = "CLF_CORRETOR"

print(f"✅ CONFIG COMP MODE_{MODE_CODE} carregada")
print("• input table  :", INFERENCIA_TABLE_FQN)
print("• mode         :", MODE_CODE, "| versão:", COMP_VERSAO)
print("• topk_ref_pct :", TOPK_REF_PCT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers

# COMMAND ----------

import math
import os
import json
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy.stats import spearmanr

# ── MLflow ────────────────────────────────────────────────────────────────────

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id

# ── I/O helpers ───────────────────────────────────────────────────────────────

def log_png(fig, artifact_path: str, dpi: int = 150):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        fig.savefig(fp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

def log_csv(df_csv: pd.DataFrame, artifact_path: str):
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, os.path.basename(artifact_path))
        df_csv.to_csv(fp, index=False)
        mlflow.log_artifact(fp, artifact_path=os.path.dirname(artifact_path) if "/" in artifact_path else None)

# ── Paletas ───────────────────────────────────────────────────────────────────

PALETTE_STATUS = {
    "Emitida":  "#A8D5BA",
    "Perdida":  "#F4B6B6",
    "_default": "#CFCFCF",
}

_MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

def model_color(i: int) -> str:
    return _MODEL_COLORS[i % len(_MODEL_COLORS)]

# ── Validação de colunas ──────────────────────────────────────────────────────

def ensure_required_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Colunas faltando no input: {missing}")

# ── Label a partir de DS_GRUPO_STATUS ────────────────────────────────────────

def derive_label(df, status_col: str = STATUS_COL):
    """Emitida → 1, demais → 0. Usa DS_GRUPO_STATUS como fonte de verdade."""
    s = F.upper(F.trim(F.col(status_col).cast("string")))
    return df.withColumn("label_real", F.when(s == F.lit("EMITIDA"), F.lit(1)).otherwise(F.lit(0)))

print("✅ Helpers carregados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparos MLflow e carga de dados

# COMMAND ----------

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
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

# Exec run (todos os logs aqui)
RUN_COMP_EXEC = run_name_vts("T_COMP")
mlflow.start_run(run_name=RUN_COMP_EXEC, nested=True)

mlflow.set_tags({
    "pipeline_tipo": "T", "stage": "COMP", "run_role": "exec",
    "mode": MODE_CODE, "comp_versao": COMP_VERSAO, "versao_ref": VERSAO_REF, "seg_target": SEG_TARGET,
})

# ── Carga e validação ─────────────────────────────────────────────────────────

df0 = spark.table(INFERENCIA_TABLE_FQN)
ensure_required_cols(df0, [STATUS_COL, DATE_COL, SEG_COL])

df0 = df0.filter(F.col(SEG_COL) == F.lit(SEG_TARGET))

# Resolver MODEL_IDS
if MODEL_IDS:
    MODEL_IDS_USED = list(MODEL_IDS)
else:
    MODEL_IDS_USED = sorted([
        c.replace("p_emitida_", "")
        for c in df0.columns
        if c.startswith("p_emitida_")
    ])

if not MODEL_IDS_USED:
    raise ValueError("❌ Nenhum model_id encontrado. Verifique a tabela ou preencha MODEL_IDS.")

p_cols  = [f"p_emitida_{m}"   for m in MODEL_IDS_USED]
rk_cols = [f"rank_global_{m}" for m in MODEL_IDS_USED]
ensure_required_cols(df0, p_cols + rk_cols)

# Derivar label a partir de DS_GRUPO_STATUS
df = derive_label(df0)
df = (
    df
    .withColumn("DATA_COTACAO_dt", F.to_date(F.col(DATE_COL)))
    .withColumn("MES", F.date_format(F.col("DATA_COTACAO_dt"), "yyyy-MM"))
    .filter(F.col("label_real").isNotNull())
    .cache()
)

N = int(df.count())
if N == 0:
    raise ValueError("❌ Sem linhas após filtros.")

n_pos     = int(df.agg(F.sum("label_real")).collect()[0][0] or 0)
base_rate = n_pos / N

HAS_CLUSTER = (MODE_CODE == "D") and (CLF_CORRETOR_COL in df.columns)
if HAS_CLUSTER:
    print(f"• CLF_CORRETOR detectado — análise por cluster será executada (Bloco 7)")
elif MODE_CODE == "D":
    print(f"⚠️  MODE_D detectado mas CLF_CORRETOR ausente — análise por cluster será pulada")

# Collect para pandas — usado em todos os blocos
_collect_cols = list(dict.fromkeys(
    ["label_real", STATUS_COL, "MES"] + p_cols + rk_cols
    + ([CLF_CORRETOR_COL] if HAS_CLUSTER else [])
))
pdf = df.select(*[c for c in _collect_cols if c in df.columns]).toPandas()

# Log params
mlflow.log_params({
    "ts_exec":              TS_EXEC,
    "comp_versao":          COMP_VERSAO,
    "versao_ref":           VERSAO_REF,
    "mode_code":            MODE_CODE,
    "seg_target":           SEG_TARGET,
    "pr_run_id":            PR_COMP_RUN_ID,
    "inferencia_table_fqn": INFERENCIA_TABLE_FQN,
    "model_ids":            json.dumps(MODEL_IDS_USED),
    "topk_step":            TOPK_STEP,
    "topk_ref_pct":         TOPK_REF_PCT,
    "n_rows":               N,
    "n_pos":                n_pos,
    "base_rate":            round(base_rate, 6),
})

K_PCTS = list(range(1, 101, TOPK_STEP))

# ── Carregar thresholds do treino (opcional) ──────────────────────────────────
client = MlflowClient()
MODEL_THRESHOLDS    = {}   # model_id → tau (float) ou None
MODEL_EVAL_SUMMARIES = {}  # model_id → full eval_summary dict (reutilizado em model_configs)

if TREINO_EXEC_RUN_ID:
    for _mid in MODEL_IDS_USED:
        try:
            with tempfile.TemporaryDirectory() as _td:
                _local = client.download_artifacts(
                    TREINO_EXEC_RUN_ID, f"eval/{_mid}/eval_summary.json", _td
                )
                with open(_local) as _f:
                    _summary = json.load(_f)
            MODEL_EVAL_SUMMARIES[_mid] = _summary
            MODEL_THRESHOLDS[_mid] = float(_summary["threshold"])
        except Exception as _e:
            print(f"⚠️  Threshold para {_mid} não carregado: {_e}")
            MODEL_THRESHOLDS[_mid] = None
    mlflow.log_param("treino_exec_run_id", TREINO_EXEC_RUN_ID)
    print("• thresholds carregados:", MODEL_THRESHOLDS)
else:
    MODEL_THRESHOLDS = {m: None for m in MODEL_IDS_USED}
    print("• TREINO_EXEC_RUN_ID não preenchido — thresholds não serão anotados")

# ── Carregar distribuição mensal de status do JOIN (opcional) ─────────────────
_pdf_status_month = None   # pd.DataFrame com colunas: MES + status cols

if JOIN_EXEC_RUN_ID:
    try:
        with tempfile.TemporaryDirectory() as _td:
            _local = client.download_artifacts(
                JOIN_EXEC_RUN_ID, "analysis/status_by_month_pivot.json", _td
            )
            with open(_local) as _f:
                _pivot_data = json.load(_f)
        if _pivot_data.get("ok"):
            _pdf_status_month = pd.DataFrame(_pivot_data["rows"])
            mlflow.log_param("join_exec_run_id", JOIN_EXEC_RUN_ID)
            print("• status-by-month carregado:", _pdf_status_month.shape)
        else:
            print(f"⚠️  status_by_month_pivot.json retornou ok=False: {_pivot_data}")
    except Exception as _e:
        print(f"⚠️  Status-by-month do JOIN não carregado: {_e}")
else:
    print("• JOIN_EXEC_RUN_ID não preenchido — barras de status não serão adicionadas ao gráfico temporal")

# ── Carregar FQNs de cotacao_model e cotacao_validacao do treino (opcional) ───
_df_model_fqn = None
_df_valid_fqn = None

if TREINO_EXEC_RUN_ID:
    try:
        _treino_params = client.get_run(TREINO_EXEC_RUN_ID).data.params
        _df_model_fqn = _treino_params.get("df_model_fqn")
        _df_valid_fqn = _treino_params.get("df_valid_fqn")
        if _df_model_fqn and _df_valid_fqn:
            print(f"• tabelas model/valid: {_df_model_fqn} | {_df_valid_fqn}")
        else:
            print("⚠️  FQNs de model/valid não encontrados nos params do treino")
    except Exception as _e:
        print(f"⚠️  FQNs de model/valid não carregados: {_e}")
else:
    print("• TREINO_EXEC_RUN_ID não preenchido — gráficos temporais de model/valid não serão gerados")

print("✅ Dados carregados")
print(f"• n_rows={N}  n_pos={n_pos}  base_rate={base_rate:.4f}")
print(f"• model_ids : {MODEL_IDS_USED}")
print(f"• PR  : {PR_COMP_NAME} ({_pr_status})")
print(f"• EXEC: {RUN_COMP_EXEC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 0 — Configurações de modelos (model_configs)

# COMMAND ----------

# feature_cols é obrigatório — requer TREINO_EXEC_RUN_ID preenchido
if not TREINO_EXEC_RUN_ID:
    raise ValueError(
        "❌ TREINO_EXEC_RUN_ID não preenchido. "
        "feature_cols é obrigatório em model_configs — preencha TREINO_EXEC_RUN_ID na Config."
    )

_treino_run_params = client.get_run(TREINO_EXEC_RUN_ID).data.params

# feature_cols: obrigatório — logado como param no T_TREINO tanto em MODE_C quanto MODE_D
_raw_feature_cols = _treino_run_params.get("feature_cols")
if not _raw_feature_cols:
    raise ValueError(
        f"❌ Param 'feature_cols' ausente no run TREINO_EXEC_RUN_ID={TREINO_EXEC_RUN_ID}. "
        "Verifique se o run_id aponta para T_TREINO (etapa=TREINO, step=TREINO)."
    )
_GLOBAL_FEATURE_COLS = json.loads(_raw_feature_cols)

# Params opcionais — compatíveis com MODE_C e MODE_D
_tp_feature_set   = _treino_run_params.get("feature_set")
_tp_pinned        = _treino_run_params.get("treino_features_pinned")
_tp_use_cw        = _treino_run_params.get("use_class_weight")
_tp_treino_versao = _treino_run_params.get("treino_versao")
_tp_eval_crit     = _treino_run_params.get("eval_criterion")
_tp_eval_prec_tgt = _treino_run_params.get("eval_precision_target")

for _mid in MODEL_IDS_USED:
    _cfg = {
        "model_id":               _mid,
        "mode_code":              MODE_CODE,
        "treino_exec_run_id":     TREINO_EXEC_RUN_ID,
        "treino_versao":          _tp_treino_versao,
        "seg_target":             _treino_run_params.get("seg_target", SEG_TARGET),
        "feature_cols":           _GLOBAL_FEATURE_COLS,
        "n_features":             len(_GLOBAL_FEATURE_COLS),
        "feature_set":            _tp_feature_set,
        "treino_features_pinned": json.loads(_tp_pinned) if _tp_pinned else [],
        "use_class_weight":       _tp_use_cw,
        "eval_criterion":         _tp_eval_crit,
        "eval_precision_target":  float(_tp_eval_prec_tgt) if _tp_eval_prec_tgt else None,
    }

    # Enriquecer com eval_summary (threshold + métricas no ponto de corte)
    _es = MODEL_EVAL_SUMMARIES.get(_mid, {})
    if _es:
        _cfg["threshold"]      = _es.get("threshold")
        _cfg["eval_precision"] = _es.get("precision")
        _cfg["eval_recall"]    = _es.get("recall")
        _cfg["eval_f1"]        = _es.get("f1")
        _cfg["eval_f2"]        = _es.get("f2")
    else:
        _cfg["threshold"] = MODEL_THRESHOLDS.get(_mid)

    # MODE_D: incluir referência ao clustering se disponível
    if MODE_CODE == "D" and CLF_FIT_EXEC_RUN_ID:
        _cfg["clf_fit_exec_run_id"] = CLF_FIT_EXEC_RUN_ID

    mlflow.log_dict(_cfg, f"model_configs/{_mid}/config.json")

print(f"✅ Bloco 0 — model_configs: {len(MODEL_IDS_USED)} config(s) logadas")
for _mid in MODEL_IDS_USED:
    print(f"  • model_configs/{_mid}/config.json  (feature_cols: {len(_GLOBAL_FEATURE_COLS)} features)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 1 — Ranking: P@K%, R@K%, Lift@K%, CM@K%

# COMMAND ----------

rows_topk = []
for model_id in MODEL_IDS_USED:
    rk_col = f"rank_global_{model_id}"
    ranks  = pdf[rk_col].values
    labels = pdf["label_real"].values

    for k_pct in K_PCTS:
        top_n = max(1, min(round(k_pct / 100.0 * N), N))
        mask  = ranks <= top_n
        tp    = int((mask & (labels == 1)).sum())
        fp    = int((mask & (labels == 0)).sum())
        fn    = int((~mask & (labels == 1)).sum())
        tn    = int((~mask & (labels == 0)).sum())

        p_at_k    = tp / top_n            if top_n > 0    else 0.0
        r_at_k    = tp / n_pos            if n_pos > 0    else 0.0
        lift_at_k = p_at_k / base_rate    if base_rate > 0 else 0.0

        rows_topk.append({
            "model_id": model_id, "k_pct": k_pct, "top_n": top_n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "p_at_k": p_at_k, "r_at_k": r_at_k, "lift_at_k": lift_at_k,
        })

pdf_topk = pd.DataFrame(rows_topk)
log_csv(pdf_topk, "ranking/topk_curves.csv")

# ── Plot: P@K%, R@K%, Lift@K% (3 subplots, modelos sobrepostos) ──────────────

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for i, model_id in enumerate(MODEL_IDS_USED):
    sub = pdf_topk[pdf_topk["model_id"] == model_id]
    c   = model_color(i)
    axs[0].plot(sub["k_pct"], sub["p_at_k"],    color=c, marker=".", markersize=2, linewidth=1.5, label=model_id)
    axs[1].plot(sub["k_pct"], sub["r_at_k"],    color=c, marker=".", markersize=2, linewidth=1.5, label=model_id)
    axs[2].plot(sub["k_pct"], sub["lift_at_k"], color=c, marker=".", markersize=2, linewidth=1.5, label=model_id)

# Baselines aleatórias
axs[0].axhline(base_rate, linestyle="--", color="gray", linewidth=1, label=f"aleatório (base_rate={base_rate:.3f})")
axs[1].plot(K_PCTS, [k / 100.0 for k in K_PCTS], linestyle="--", color="gray", linewidth=1, label="aleatório")
axs[2].axhline(1.0, linestyle="--", color="gray", linewidth=1, label="aleatório")

for ax, ylabel in zip(axs, ["Precision@K", "Recall@K", "Lift@K"]):
    ax.set_ylabel(ylabel); ax.grid(True); ax.legend()
axs[2].set_xlabel("Top-K (%)")

fig.suptitle(f"Ranking — P@K%, R@K%, Lift@K% por modelo (MODE_C | {SEG_TARGET})")
plt.tight_layout()
log_png(fig, "ranking/curves_ranking.png")

# ── Plot: CM@K% — 4 subplots (TP, FP, FN, TN) ───────────────────────────────

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
cm_def = [("tp", "TP@K%"), ("fp", "FP@K%"), ("fn", "FN@K%"), ("tn", "TN@K%")]
ax_flat = axs.flatten()

for ax_i, (metric, label) in enumerate(cm_def):
    for i, model_id in enumerate(MODEL_IDS_USED):
        sub = pdf_topk[pdf_topk["model_id"] == model_id]
        ax_flat[ax_i].plot(sub["k_pct"], sub[metric], color=model_color(i),
                           marker=".", markersize=2, linewidth=1.5, label=model_id)
    ax_flat[ax_i].set_title(label)
    ax_flat[ax_i].set_ylabel("contagem")
    ax_flat[ax_i].grid(True)
    ax_flat[ax_i].legend()

for ax_i in [2, 3]:
    ax_flat[ax_i].set_xlabel("Top-K (%)")

fig.suptitle(f"Confusion Matrix @K% por modelo (MODE_C | {SEG_TARGET})")
plt.tight_layout()
log_png(fig, "ranking/curves_cm_at_k.png")

print("✅ Bloco 1 — Ranking concluído")
print(f"• topk_curves.csv: {len(pdf_topk)} linhas ({len(MODEL_IDS_USED)} modelos × {len(K_PCTS)} K%s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 2 — Classificação geral: AUC-PR e AP

# COMMAND ----------

auc_pr_summary = {}
pr_curves_data = {}   # model_id → {precision, recall, thresholds, pdf_curve}

for model_id in MODEL_IDS_USED:
    p_col   = f"p_emitida_{model_id}"
    valid   = pdf[p_col].notna()
    y_score = pdf.loc[valid, p_col].astype(float).values
    y_true  = pdf.loc[valid, "label_real"].astype(int).values

    ap              = float(average_precision_score(y_true, y_score))
    prec, rec, thr  = precision_recall_curve(y_true, y_score)
    aucp            = float(np.trapz(prec[::-1], rec[::-1]))

    # DataFrame de curva para threshold_metrics (exclui ponto sentinela final)
    _pdf_curve = pd.DataFrame({
        "threshold": thr,
        "precision": prec[:-1],
        "recall":    rec[:-1],
    }).sort_values("threshold").reset_index(drop=True)
    _pdf_curve["f1"] = (2 * _pdf_curve["precision"] * _pdf_curve["recall"] /
                        (_pdf_curve["precision"] + _pdf_curve["recall"] + 1e-10))
    _pdf_curve["f2"] = (5 * _pdf_curve["precision"] * _pdf_curve["recall"] /
                        (4 * _pdf_curve["precision"] + _pdf_curve["recall"] + 1e-10))

    auc_pr_summary[model_id] = {"ap": ap, "auc_pr": aucp, "base_rate": round(base_rate, 6)}
    pr_curves_data[model_id] = {
        "precision": prec.tolist(), "recall": rec.tolist(), "thresholds": thr.tolist(),
        "pdf_curve": _pdf_curve,
    }
    mlflow.log_metrics({f"ap_{model_id}": round(ap, 6), f"auc_pr_{model_id}": round(aucp, 6)})

mlflow.log_dict(auc_pr_summary, "classification/auc_pr_summary.json")

# ── Plot: curvas PR sobrepostas com anotação de threshold ────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
ax.axhline(base_rate, linestyle="--", color="gray", linewidth=1,
           label=f"no-skill (base_rate={base_rate:.3f})")

for i, model_id in enumerate(MODEL_IDS_USED):
    prec = pr_curves_data[model_id]["precision"]
    rec  = pr_curves_data[model_id]["recall"]
    thr  = np.array(pr_curves_data[model_id]["thresholds"])
    ap   = auc_pr_summary[model_id]["ap"]
    c    = model_color(i)

    ax.plot(rec, prec, color=c, linewidth=1.5, label=f"{model_id} (AP={ap:.4f})")

    tau = MODEL_THRESHOLDS.get(model_id)
    if tau is not None and len(thr) > 0:
        idx          = int(np.argmin(np.abs(thr - tau)))
        tau_rec      = rec[idx]
        tau_prec     = prec[idx]
        ax.scatter([tau_rec], [tau_prec], color=c, s=80, zorder=5)
        ax.annotate(f"τ={tau:.2f}", xy=(tau_rec, tau_prec),
                    xytext=(tau_rec + 0.02, tau_prec - 0.04),
                    fontsize=8, color=c,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title(f"Curvas Precision-Recall por modelo (MODE_C | {SEG_TARGET})")
ax.grid(True); ax.legend()
plt.tight_layout()
log_png(fig, "classification/pr_curves.png")

print("✅ Bloco 2 — Classificação concluído")
for m, v in auc_pr_summary.items():
    print(f"  • {m}: AP={v['ap']:.4f}  AUC-PR={v['auc_pr']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 2b — Métricas por threshold (P, R, F1, F2 vs τ)

# COMMAND ----------

n_mod_b = len(MODEL_IDS_USED)
fig, axs_b = plt.subplots(n_mod_b, 1, figsize=(9, 4 * n_mod_b), squeeze=False)

for i, model_id in enumerate(MODEL_IDS_USED):
    _df = pr_curves_data[model_id]["pdf_curve"].dropna(subset=["precision", "recall"])
    ax  = axs_b[i][0]

    ax.plot(_df["threshold"], _df["precision"], label="Precision", color="steelblue",  lw=2)
    ax.plot(_df["threshold"], _df["recall"],    label="Recall",    color="darkorange", lw=2)
    ax.plot(_df["threshold"], _df["f1"],        label="F1",        color="seagreen",   lw=2)
    ax.plot(_df["threshold"], _df["f2"],        label="F2",        color="mediumpurple", lw=2)

    tau = MODEL_THRESHOLDS.get(model_id)
    if tau is not None:
        ax.axvline(tau, color="crimson", linestyle="--", lw=1.5, label=f"τ treino = {tau:.2f}")

    ax.set_title(f"Métricas por threshold — {model_id}")
    ax.set_xlabel("Threshold (τ)"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True); ax.legend(fontsize=9)

fig.suptitle(f"Threshold metrics por modelo (MODE_C | {SEG_TARGET})")
plt.tight_layout()
log_png(fig, "classification/threshold_metrics.png")

print("✅ Bloco 2b — Threshold metrics concluído")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 2c — Overfitting: treino vs validação

# COMMAND ----------

if TREINO_EXEC_RUN_ID:
    _treino_metrics = client.get_run(TREINO_EXEC_RUN_ID).data.metrics

    overfitting_rows = []
    for model_id in MODEL_IDS_USED:
        auc_pr_tr = _treino_metrics.get(f"auc_pr_treino_{model_id}")
        ap_tr     = _treino_metrics.get(f"ap_treino_{model_id}")
        auc_pr_vl = auc_pr_summary[model_id]["auc_pr"]
        ap_vl     = auc_pr_summary[model_id]["ap"]

        overfitting_rows.append({
            "model_id":    model_id,
            "auc_pr_treino": round(float(auc_pr_tr), 6) if auc_pr_tr is not None else None,
            "auc_pr_val":    round(float(auc_pr_vl), 6),
            "gap_auc_pr":    round(float(auc_pr_tr) - float(auc_pr_vl), 6) if auc_pr_tr is not None else None,
            "ap_treino":     round(float(ap_tr), 6) if ap_tr is not None else None,
            "ap_val":        round(float(ap_vl), 6),
            "gap_ap":        round(float(ap_tr) - float(ap_vl), 6) if ap_tr is not None else None,
        })

    pdf_overfit = pd.DataFrame(overfitting_rows)
    log_csv(pdf_overfit, "overfitting/overfitting_summary.csv")
    mlflow.log_dict(pdf_overfit.to_dict(orient="records"), "overfitting/overfitting_summary.json")

    # ── Plot: barras agrupadas treino vs validação ────────────────────────────────
    x      = np.arange(len(MODEL_IDS_USED))
    bw_bar = 0.35
    fig, axs_ov = plt.subplots(1, 2, figsize=(12, 5))

    for ax_i, (metric_tr, metric_vl, title) in enumerate([
        ("auc_pr_treino", "auc_pr_val", "AUC-PR"),
        ("ap_treino",     "ap_val",     "Average Precision (AP)"),
    ]):
        ax = axs_ov[ax_i]
        vals_tr = pdf_overfit[metric_tr].fillna(0).values
        vals_vl = pdf_overfit[metric_vl].values

        ax.bar(x - bw_bar / 2, vals_tr, bw_bar, label="Treino",    color="#A8C8E8", edgecolor="white")
        ax.bar(x + bw_bar / 2, vals_vl, bw_bar, label="Validação", color="#A8D5BA", edgecolor="white")

        # Anotar gap
        for j, row in pdf_overfit.iterrows():
            gap = row[f"gap_{metric_tr.split('_treino')[0]}"] if metric_tr != "ap_treino" \
                  else row["gap_ap"]
            if gap is not None:
                ax.text(j, max(vals_tr[j], vals_vl[j]) + 0.01,
                        f"Δ{gap:+.3f}", ha="center", va="bottom", fontsize=8,
                        color="crimson" if gap > 0.05 else "gray")

        ax.set_title(title)
        ax.set_xticks(x); ax.set_xticklabels(MODEL_IDS_USED, rotation=15, ha="right")
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y"); ax.legend()

    fig.suptitle(f"Overfitting — Treino vs Validação por modelo (MODE_C | {SEG_TARGET})")
    plt.tight_layout()
    log_png(fig, "overfitting/overfitting_comparison.png")

    print("✅ Bloco 2c — Overfitting concluído")
    print(pdf_overfit.to_string(index=False))
else:
    print("⚠️  Bloco 2c ignorado — TREINO_EXEC_RUN_ID não preenchido")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 3 — Distribuição de scores

# COMMAND ----------

# Normaliza status para estratificação (fonte: DS_GRUPO_STATUS)
pdf["status_simple"] = (
    pdf[STATUS_COL].str.strip().str.upper()
    .map({"EMITIDA": "Emitida", "PERDIDA": "Perdida"})
    .fillna("Outros")
)

bins    = np.linspace(0, 1, 51)
centers = (bins[:-1] + bins[1:]) / 2.0
bw      = bins[1] - bins[0]
n_mod   = len(MODEL_IDS_USED)

# Layout: 3 subplots por linha (apenas densidade)
_ncols = 3
_nrows = math.ceil(n_mod / _ncols)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 4 * _nrows), squeeze=False)

score_stats = {}

for i, model_id in enumerate(MODEL_IDS_USED):
    p_col   = f"p_emitida_{model_id}"
    ax      = axes[i // _ncols][i % _ncols]
    stats_m = {}

    for st in ["Emitida", "Perdida"]:
        vals = pdf.loc[pdf["status_simple"] == st, p_col].dropna().astype(float).values
        if len(vals) == 0:
            continue
        hist_vals, _ = np.histogram(vals, bins=bins, density=True)
        ax.bar(centers, hist_vals, width=bw, alpha=0.5,
               label=st, color=PALETTE_STATUS.get(st, PALETTE_STATUS["_default"]),
               edgecolor="white", linewidth=0.5)

    ax.set_title(model_id)
    ax.set_xlabel("p_emitida")
    ax.set_ylabel("densidade")
    ax.grid(True, axis="y")
    ax.legend()

    # Estatísticas por label
    for label_val, label_name in [(1, "emitida"), (0, "perdida")]:
        vals = pdf.loc[pdf["label_real"] == label_val, p_col].dropna().astype(float)
        stats_m[label_name] = {
            "n":    len(vals),
            "mean": round(float(vals.mean()), 6) if len(vals) > 0 else None,
            "std":  round(float(vals.std()),  6) if len(vals) > 0 else None,
            "min":  round(float(vals.min()),  6) if len(vals) > 0 else None,
            "max":  round(float(vals.max()),  6) if len(vals) > 0 else None,
        }
    score_stats[model_id] = stats_m

# Esconder subplots não utilizados
for j in range(n_mod, _nrows * _ncols):
    axes[j // _ncols][j % _ncols].set_visible(False)

fig.suptitle(f"Distribuição de scores por DS_GRUPO_STATUS (MODE_C | {SEG_TARGET})")
plt.tight_layout()
log_png(fig, "scores/score_distribution.png")
mlflow.log_dict(score_stats, "scores/score_stats.json")

print("✅ Bloco 3 — Distribuição de scores concluído")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 4 — Concordância entre modelos

# COMMAND ----------

if len(MODEL_IDS_USED) > 1:
    # ── Correlação de Spearman entre rank_global ──────────────────────────────────

    n_m = len(MODEL_IDS_USED)
    corr_matrix = np.eye(n_m)
    for r in range(n_m):
        for c in range(r + 1, n_m):
            rho, _ = spearmanr(
                pdf[f"rank_global_{MODEL_IDS_USED[r]}"],
                pdf[f"rank_global_{MODEL_IDS_USED[c]}"],
            )
            corr_matrix[r][c] = corr_matrix[c][r] = float(rho)

    corr_dict = {
        MODEL_IDS_USED[r]: {MODEL_IDS_USED[c]: round(corr_matrix[r][c], 6) for c in range(n_m)}
        for r in range(n_m)
    }
    mlflow.log_dict(corr_dict, "concordance/rank_correlation.json")

    # Plot heatmap
    fig_sz = max(5, n_m + 1)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_m)); ax.set_xticklabels(MODEL_IDS_USED, rotation=45, ha="right")
    ax.set_yticks(range(n_m)); ax.set_yticklabels(MODEL_IDS_USED)
    for r in range(n_m):
        for c in range(n_m):
            ax.text(c, r, f"{corr_matrix[r][c]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Correlação de Spearman entre rank_global (MODE_C | {SEG_TARGET})")
    plt.tight_layout()
    log_png(fig, "concordance/rank_correlation_heatmap.png")

    # ── Overlap@K% ────────────────────────────────────────────────────────────────

    overlap_rows = []
    for k_pct in K_PCTS:
        top_n = max(1, min(round(k_pct / 100.0 * N), N))
        sets  = [set(pdf.index[pdf[f"rank_global_{m}"] <= top_n]) for m in MODEL_IDS_USED]
        inter = set.intersection(*sets)
        overlap_rows.append({
            "k_pct": k_pct, "top_n": top_n,
            "n_overlap": len(inter),
            "overlap_pct": len(inter) / top_n,
        })

    pdf_overlap = pd.DataFrame(overlap_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pdf_overlap["k_pct"], pdf_overlap["overlap_pct"] * 100,
            color="#1f77b4", linewidth=1.5, marker=".", markersize=2)
    ax.set_xlabel("Top-K (%)"); ax.set_ylabel("% de overlap (todos os modelos)")
    ax.set_title(f"Overlap@K% entre todos os modelos (MODE_C | {SEG_TARGET})")
    ax.grid(True)
    plt.tight_layout()
    log_png(fig, "concordance/overlap_at_k.png")

    print("✅ Bloco 4 — Concordância concluído")
    print(f"• Correlações de Spearman:\n{pd.DataFrame(corr_dict).round(3).to_string()}")
else:
    print("⚠️  Bloco 4 ignorado — apenas 1 modelo (requer ≥ 2 modelos)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 5 — Distribuição temporal de cotações (model / validação)

# COMMAND ----------

def _plot_cotacao_by_date(table_fqn: str, label: str, artifact_name: str):
    """Gera gráfico de barras: contagem de cotações por DATA_COTACAO."""
    _df = spark.read.table(table_fqn)
    _df_agg = (
        _df.withColumn("DATA_COTACAO_dt", F.to_date(F.col(DATE_COL)))
        .filter(F.col("DATA_COTACAO_dt").isNotNull())
        .groupBy("DATA_COTACAO_dt")
        .agg(F.count("*").alias("qtd"))
        .orderBy("DATA_COTACAO_dt")
    )
    _pdf_agg = _df_agg.toPandas()

    if _pdf_agg.empty:
        print(f"⚠️  {label}: sem dados de DATA_COTACAO para plotar")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        _pdf_agg["DATA_COTACAO_dt"].astype(str),
        _pdf_agg["qtd"],
        color="#4A90D9", edgecolor="white", linewidth=0.3,
    )
    ax.set_xlabel("DATA_COTACAO")
    ax.set_ylabel("Qtd cotações")
    ax.set_title(f"{label} — Contagem por DATA_COTACAO ({SEG_TARGET})\n{table_fqn}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    log_png(fig, f"temporal/{artifact_name}")
    print(f"  • {artifact_name} gerado ({len(_pdf_agg)} datas)")


if _df_model_fqn and _df_valid_fqn:
    _plot_cotacao_by_date(_df_model_fqn, "cotacao_model", "cotacao_model_by_date.png")
    _plot_cotacao_by_date(_df_valid_fqn, "cotacao_validacao", "cotacao_validacao_by_date.png")
    print(f"✅ Bloco 5 — Distribuição temporal concluída")
else:
    print("⚠️  Bloco 5 ignorado — FQNs de model/valid não disponíveis (preencha TREINO_EXEC_RUN_ID)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 6 — Tabela de seleção

# COMMAND ----------

# K de referência: usa TOPK_REF_PCT (ou o K mais próximo disponível)
k_ref_avail = min(K_PCTS, key=lambda x: abs(x - TOPK_REF_PCT))

selection_rows = []
for model_id in MODEL_IDS_USED:
    sub_ref = pdf_topk[(pdf_topk["model_id"] == model_id) & (pdf_topk["k_pct"] == k_ref_avail)]

    row = {
        "model_id":                  model_id,
        "auc_pr":                    round(auc_pr_summary[model_id]["auc_pr"], 6),
        "ap":                        round(auc_pr_summary[model_id]["ap"],    6),
        f"p_at_{k_ref_avail}pct":    round(float(sub_ref["p_at_k"].iloc[0]),    6) if len(sub_ref) else None,
        f"r_at_{k_ref_avail}pct":    round(float(sub_ref["r_at_k"].iloc[0]),    6) if len(sub_ref) else None,
        f"lift_at_{k_ref_avail}pct": round(float(sub_ref["lift_at_k"].iloc[0]), 6) if len(sub_ref) else None,
    }
    selection_rows.append(row)

pdf_selection = pd.DataFrame(selection_rows)
log_csv(pdf_selection, "summary/model_selection_table.csv")
mlflow.log_dict(pdf_selection.to_dict(orient="records"), "summary/model_selection_table.json")

print(f"✅ Bloco 6 — Tabela de seleção (K_ref={k_ref_avail}%)")
print(pdf_selection.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bloco 7 — Análise por cluster (apenas MODE_D)

# COMMAND ----------

if not HAS_CLUSTER:
    mlflow.set_tag("cluster_analysis", "skipped")
    print("⚠️  Bloco 7 ignorado — HAS_CLUSTER=False (MODE_CODE != D ou CLF_CORRETOR ausente na tabela de inferência)")
else:
    mlflow.set_tag("cluster_analysis", "executed")

    # ── Carregar cluster_summary do T_CLUSTERING_FIT (opcional) ──────────────────
    _clf_summary_by_id = {}   # cluster_id (str) → dict com centroides e n_corretores

    if CLF_FIT_EXEC_RUN_ID:
        try:
            with tempfile.TemporaryDirectory() as _td:
                _local = client.download_artifacts(
                    CLF_FIT_EXEC_RUN_ID, "clustering/cluster_summary.json", _td
                )
                with open(_local) as _f:
                    _clf_summary_list = json.load(_f)
            for _cs in _clf_summary_list:
                _clf_summary_by_id[str(_cs["cluster"])] = _cs
            mlflow.log_param("clf_fit_exec_run_id", CLF_FIT_EXEC_RUN_ID)
            print(f"• cluster_summary carregado: {len(_clf_summary_by_id)} clusters")
        except Exception as _e:
            print(f"⚠️  cluster_summary não carregado: {_e}")
    else:
        print("• CLF_FIT_EXEC_RUN_ID não preenchido — centroides não incluídos em cluster_profile.json")

    # Paleta de cores dedicada para clusters
    _CLF_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ]
    def _clf_color(i): return _CLF_COLORS[i % len(_CLF_COLORS)]

    # Clusters presentes em pdf (ordenados numericamente)
    _cluster_ids = sorted(pdf[CLF_CORRETOR_COL].dropna().unique().tolist(), key=lambda x: int(x))
    print(f"• clusters em df_inf: {_cluster_ids}")

    # ── Bloco 7a: cluster_profile.json + métricas de classificação por cluster ────

    _clf_profile_rows = []
    _clf_metrics_rows = []   # acumula para metrics_summary.csv

    for c_id in _cluster_ids:
        mask_c  = pdf[CLF_CORRETOR_COL] == c_id
        pdf_c   = pdf[mask_c].copy()
        n_c     = len(pdf_c)
        n_pos_c = int((pdf_c["label_real"] == 1).sum())
        br_c    = n_pos_c / n_c if n_c > 0 else 0.0

        # cluster_profile enriquecido com centroides se disponíveis
        _cs_entry = _clf_summary_by_id.get(str(c_id), {})
        _profile_row = {
            "cluster":     c_id,
            "n_cotacoes":  n_c,
            "n_positivos": n_pos_c,
            "base_rate":   round(br_c, 6),
        }
        if _cs_entry:
            _profile_row["n_corretores"] = _cs_entry.get("n_corretores")
            for _k, _v in _cs_entry.items():
                if _k not in ("cluster", "n_corretores"):
                    _profile_row[_k] = _v
        _clf_profile_rows.append(_profile_row)

        # Métricas de volume/base_rate logadas por cluster (independente de modelo)
        mlflow.log_metrics({
            f"n_cotacoes_cluster_{c_id}":  n_c,
            f"n_positivos_cluster_{c_id}": n_pos_c,
            f"base_rate_cluster_{c_id}":   round(br_c, 6),
        })

        # Métricas de classificação por modelo
        for model_id in MODEL_IDS_USED:
            p_col  = f"p_emitida_{model_id}"
            _valid = pdf_c[p_col].notna()
            ap_c = aucp_c = None
            if _valid.sum() >= 2 and n_pos_c > 0:
                _ys = pdf_c.loc[_valid, p_col].astype(float).values
                _yt = pdf_c.loc[_valid, "label_real"].astype(int).values
                if len(np.unique(_yt)) >= 2:
                    ap_c   = float(average_precision_score(_yt, _ys))
                    _prec_c, _rec_c, _ = precision_recall_curve(_yt, _ys)
                    aucp_c = float(np.trapz(_prec_c[::-1], _rec_c[::-1]))
                    mlflow.log_metrics({
                        f"ap_{model_id}_cluster_{c_id}":     round(ap_c, 6),
                        f"auc_pr_{model_id}_cluster_{c_id}": round(aucp_c, 6),
                    })

            _clf_metrics_rows.append({
                "cluster":   c_id,
                "model_id":  model_id,
                "n":         n_c,
                "n_pos":     n_pos_c,
                "base_rate": round(br_c, 6),
                "ap":        round(ap_c, 6) if ap_c is not None else None,
                "auc_pr":    round(aucp_c, 6) if aucp_c is not None else None,
            })

    mlflow.log_dict(_clf_profile_rows, "clustering/cluster_profile.json")
    _pdf_clf_metrics = pd.DataFrame(_clf_metrics_rows)
    log_csv(_pdf_clf_metrics, "clustering/metrics_summary.csv")

    print(f"✅ Bloco 7a — cluster_profile e métricas de classificação ({len(_cluster_ids)} clusters)")

    # ── Bloco 7b: bar charts AP e AUC-PR por cluster ─────────────────────────────

    _n_models_c = len(MODEL_IDS_USED)
    _bar_x_c    = np.arange(len(_cluster_ids))
    _bw_c       = 0.8 / _n_models_c

    for _metric_c, _ylabel_c, _out_c in [
        ("ap",     "Average Precision (AP)", "ap_by_cluster.png"),
        ("auc_pr", "AUC-PR",                 "auc_pr_by_cluster.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, len(_cluster_ids)), 5))
        for _mi, model_id in enumerate(MODEL_IDS_USED):
            _sub_c = _pdf_clf_metrics[_pdf_clf_metrics["model_id"] == model_id].set_index("cluster")
            _vals_c = [
                float(_sub_c.loc[c_id, _metric_c])
                if (c_id in _sub_c.index and _sub_c.loc[c_id, _metric_c] is not None
                    and not pd.isna(_sub_c.loc[c_id, _metric_c]))
                else 0.0
                for c_id in _cluster_ids
            ]
            _offsets_c = _bar_x_c + (_mi - (_n_models_c - 1) / 2.0) * _bw_c
            ax.bar(_offsets_c, _vals_c, _bw_c, label=model_id,
                   color=model_color(_mi), edgecolor="white", alpha=0.85)

        ax.set_xticks(_bar_x_c)
        ax.set_xticklabels([f"C{c}" for c in _cluster_ids])
        ax.set_xlabel("Cluster"); ax.set_ylabel(_ylabel_c)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{_ylabel_c} por cluster (MODE_D | {SEG_TARGET})")
        ax.grid(True, axis="y"); ax.legend()
        plt.tight_layout()
        log_png(fig, f"clustering/{_out_c}")

    print("✅ Bloco 7b — bar charts AP/AUC-PR por cluster")

    # ── Bloco 7c: PR curves sobrepostas por cluster (1 plot por modelo) ──────────

    for model_id in MODEL_IDS_USED:
        p_col = f"p_emitida_{model_id}"
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.axhline(base_rate, linestyle="--", color="gray", linewidth=1,
                   label=f"no-skill global (base_rate={base_rate:.3f})")

        for _ci, c_id in enumerate(_cluster_ids):
            _mask_c = pdf[CLF_CORRETOR_COL] == c_id
            _pdf_c  = pdf[_mask_c]
            _valid  = _pdf_c[p_col].notna()
            if _valid.sum() < 2:
                continue
            _ys = _pdf_c.loc[_valid, p_col].astype(float).values
            _yt = _pdf_c.loc[_valid, "label_real"].astype(int).values
            if len(np.unique(_yt)) < 2:
                continue
            _prec_c, _rec_c, _ = precision_recall_curve(_yt, _ys)
            _ap_c = float(average_precision_score(_yt, _ys))
            ax.plot(_rec_c, _prec_c, color=_clf_color(_ci), linewidth=1.5,
                    label=f"C{c_id} (AP={_ap_c:.4f}, n={int(_mask_c.sum())})")

        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR Curves por cluster — {model_id} (MODE_D | {SEG_TARGET})")
        ax.grid(True); ax.legend(fontsize=8)
        plt.tight_layout()
        log_png(fig, f"clustering/pr_curves_by_cluster_{model_id}.png")

    print("✅ Bloco 7c — PR curves por cluster")

    # ── Bloco 7d: Top-K por cluster + Lift@K% ────────────────────────────────────

    _topk_clf_rows = []

    for c_id in _cluster_ids:
        _mask_c  = pdf[CLF_CORRETOR_COL] == c_id
        _pdf_c   = pdf[_mask_c].copy().reset_index(drop=True)
        _n_c     = len(_pdf_c)
        _n_pos_c = int((_pdf_c["label_real"] == 1).sum())
        if _n_c == 0 or _n_pos_c == 0:
            continue
        _br_c = _n_pos_c / _n_c

        for model_id in MODEL_IDS_USED:
            p_col    = f"p_emitida_{model_id}"
            _pdf_c_s = _pdf_c.sort_values(p_col, ascending=False).reset_index(drop=True)
            _pdf_c_s["_rank_local_c"] = range(1, _n_c + 1)

            for k_pct in K_PCTS:
                _top_n_c = max(1, min(round(k_pct / 100.0 * _n_c), _n_c))
                _mask_k  = _pdf_c_s["_rank_local_c"] <= _top_n_c
                _tp_c    = int((_mask_k & (_pdf_c_s["label_real"] == 1)).sum())
                _p_k_c   = _tp_c / _top_n_c if _top_n_c > 0 else 0.0
                _r_k_c   = _tp_c / _n_pos_c if _n_pos_c > 0 else 0.0
                _lift_c  = _p_k_c / _br_c   if _br_c > 0    else 0.0

                _topk_clf_rows.append({
                    "cluster":    c_id,
                    "model_id":   model_id,
                    "k_pct":      k_pct,
                    "top_n":      _top_n_c,
                    "tp":         _tp_c,
                    "p_at_k":     _p_k_c,
                    "r_at_k":     _r_k_c,
                    "lift_at_k":  _lift_c,
                })

    if _topk_clf_rows:
        _pdf_topk_clf = pd.DataFrame(_topk_clf_rows)
        log_csv(_pdf_topk_clf, "clustering/topk_curves_by_cluster.csv")

        for model_id in MODEL_IDS_USED:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="aleatório")
            for _ci, c_id in enumerate(_cluster_ids):
                _sub = _pdf_topk_clf[
                    (_pdf_topk_clf["cluster"] == c_id) & (_pdf_topk_clf["model_id"] == model_id)
                ]
                if _sub.empty:
                    continue
                ax.plot(_sub["k_pct"], _sub["lift_at_k"], color=_clf_color(_ci),
                        marker=".", markersize=2, linewidth=1.5, label=f"C{c_id}")
            ax.set_xlabel("Top-K (%)"); ax.set_ylabel("Lift@K")
            ax.set_title(f"Lift@K% por cluster — {model_id} (MODE_D | {SEG_TARGET})")
            ax.grid(True); ax.legend(fontsize=8)
            plt.tight_layout()
            log_png(fig, f"clustering/lift_by_cluster_{model_id}.png")

    print(f"✅ Bloco 7d — Top-K por cluster ({len(_topk_clf_rows)} linhas)")

    # ── Bloco 7e: Distribuição de scores por cluster ──────────────────────────────

    _bins_c    = np.linspace(0, 1, 51)
    _centers_c = (_bins_c[:-1] + _bins_c[1:]) / 2.0
    _bw_hist_c = _bins_c[1] - _bins_c[0]

    _ncols_c = 3
    _nrows_c = math.ceil(len(MODEL_IDS_USED) / _ncols_c)
    fig, _axes_c = plt.subplots(_nrows_c, _ncols_c, figsize=(5 * _ncols_c, 4 * _nrows_c), squeeze=False)

    _score_stats_c = {}   # model_id → {cluster_id → {n, mean, std, p25, p50, p75, p90}}

    for _mi, model_id in enumerate(MODEL_IDS_USED):
        p_col    = f"p_emitida_{model_id}"
        _ax_c    = _axes_c[_mi // _ncols_c][_mi % _ncols_c]
        _stats_c = {}

        for _ci, c_id in enumerate(_cluster_ids):
            _mask_c = pdf[CLF_CORRETOR_COL] == c_id
            _vals_c = pdf.loc[_mask_c, p_col].dropna().astype(float).values
            if len(_vals_c) == 0:
                continue
            _hist_v, _ = np.histogram(_vals_c, bins=_bins_c, density=True)
            _ax_c.plot(_centers_c, _hist_v, color=_clf_color(_ci),
                       linewidth=1.5, label=f"C{c_id}", alpha=0.85)
            _stats_c[c_id] = {
                "n":   len(_vals_c),
                "mean": round(float(np.mean(_vals_c)),         6),
                "std":  round(float(np.std(_vals_c)),          6),
                "p25":  round(float(np.percentile(_vals_c, 25)), 6),
                "p50":  round(float(np.percentile(_vals_c, 50)), 6),
                "p75":  round(float(np.percentile(_vals_c, 75)), 6),
                "p90":  round(float(np.percentile(_vals_c, 90)), 6),
            }

        _ax_c.set_title(model_id)
        _ax_c.set_xlabel("p_emitida"); _ax_c.set_ylabel("densidade")
        _ax_c.grid(True, axis="y"); _ax_c.legend(fontsize=8)
        _score_stats_c[model_id] = _stats_c

    for _ji in range(len(MODEL_IDS_USED), _nrows_c * _ncols_c):
        _axes_c[_ji // _ncols_c][_ji % _ncols_c].set_visible(False)

    fig.suptitle(f"Distribuição de scores por cluster (MODE_D | {SEG_TARGET})")
    plt.tight_layout()
    log_png(fig, "clustering/score_distribution_by_cluster.png")
    mlflow.log_dict(_score_stats_c, "clustering/score_stats_by_cluster.json")

    print("✅ Bloco 7e — Distribuição de scores por cluster")
    print(f"\n✅ Bloco 7 completo — {len(_cluster_ids)} clusters × {len(MODEL_IDS_USED)} modelos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encerramento

# COMMAND ----------

df.unpersist()

while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas")
print(f"\n• COMP run        : {RUN_COMP_EXEC}")
print(f"• Tabela analisada: {INFERENCIA_TABLE_FQN}")
print(f"• Modelos         : {MODEL_IDS_USED}")
print(f"\n⚠️  Artifacts no MLflow run '{RUN_COMP_EXEC}':")
print("  model_configs/  → {model_id}/config.json  (feature_cols, threshold, eval_summary, treino params)")
print("  ranking/        → topk_curves.csv, curves_ranking.png, curves_cm_at_k.png")
print("  classification/ → pr_curves.png, threshold_metrics.png, auc_pr_summary.json")
print("  overfitting/    → overfitting_summary.csv, overfitting_comparison.png")
print("  scores/         → score_distribution.png, score_stats.json")
print("  concordance/    → rank_correlation_heatmap.png, overlap_at_k.png, rank_correlation.json")
print("  temporal/       → precision_monthly.png, monthly_stats.csv")
print("  summary/        → model_selection_table.csv, model_selection_table.json")
if HAS_CLUSTER:
    print("  clustering/     → cluster_profile.json, metrics_summary.csv")
    print("                    ap_by_cluster.png, auc_pr_by_cluster.png")
    print("                    pr_curves_by_cluster_{model_id}.png")
    print("                    topk_curves_by_cluster.csv, lift_by_cluster_{model_id}.png")
    print("                    score_distribution_by_cluster.png, score_stats_by_cluster.json")