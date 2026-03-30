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
EXPERIMENT_NAME  = "/Users/psw.service@pswdigital.com.br/ISA_DEV/ISA_EXP"
PR_REPORT_NAME   = "T_PR_REPORT"
MODE_CODE        = "C"
REPORT_VERSAO    = "V1.0.0"
VERSAO_REF       = REPORT_VERSAO

TS_EXEC    = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
RUN_SUFFIX = TS_EXEC

def run_name_vts(base: str) -> str:
    return f"{base}_{TS_EXEC}"

# Override: preencha para reutilizar container PR existente
PR_RUN_ID_OVERRIDE = ""

# =========================
# INPUTS — run IDs de referência
# =========================
# run_id do exec run de 1_PRE_PROC (etapa PRE_PROC)
PRE_PROC_EXEC_RUN_ID       = ""   # <<< AJUSTE

# run_id do exec run T_PRE_PROC_MODEL_YYYYMMDD_HHMMSS (dentro de 3_TREINO)
PRE_PROC_MODEL_EXEC_RUN_ID = ""   # <<< AJUSTE

# run_id do exec run T_FEATURE_SELECTION_YYYYMMDD_HHMMSS
FS_EXEC_RUN_ID             = ""   # <<< AJUSTE

# run_id do exec run T_COMP_YYYYMMDD_HHMMSS
COMP_EXEC_RUN_ID           = ""   # <<< AJUSTE

print("✅ CONFIG REPORT MODE_C carregada")
print("• mode                     :", MODE_CODE, "| versão:", REPORT_VERSAO)
print("• pre_proc_exec_run_id     :", PRE_PROC_EXEC_RUN_ID)
print("• pre_proc_model_exec_run_id:", PRE_PROC_MODEL_EXEC_RUN_ID)
print("• fs_exec_run_id           :", FS_EXEC_RUN_ID)
print("• comp_exec_run_id         :", COMP_EXEC_RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports e helpers

# COMMAND ----------

import os
import json
import shutil
import tempfile

from mlflow.tracking import MlflowClient

def mlflow_get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abre runs MLflow (PR + exec)

# COMMAND ----------

_ = mlflow_get_or_create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
while mlflow.active_run() is not None:
    mlflow.end_run()

# PR container (sem logs)
if PR_RUN_ID_OVERRIDE:
    mlflow.start_run(run_id=PR_RUN_ID_OVERRIDE)
    PR_RUN_ID  = PR_RUN_ID_OVERRIDE
    _pr_status = "acoplada (override)"
else:
    mlflow.start_run(run_name=PR_REPORT_NAME)
    PR_RUN_ID  = mlflow.active_run().info.run_id
    _pr_status = "nova"

# Exec run (todos os logs aqui)
RUN_REPORT_EXEC = run_name_vts("T_REPORT")
mlflow.start_run(run_name=RUN_REPORT_EXEC, nested=True)

mlflow.set_tags({
    "pipeline_tipo": "T",
    "stage":         "REPORT",
    "run_role":      "exec",
    "mode":          MODE_CODE,
    "versao_ref":    VERSAO_REF,
})

print(f"✅ Runs abertas | PR {_pr_status}")
print(f"• PR_RUN_ID       : {PR_RUN_ID}")
print(f"• exec run name   : {RUN_REPORT_EXEC}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carrega metadados das runs de referência

# COMMAND ----------

client = MlflowClient()

# ── Params do COMP run ────────────────────────────────────────────────────────
_comp_run    = client.get_run(COMP_EXEC_RUN_ID)
_comp_params = _comp_run.data.params

SEG_TARGET         = _comp_params.get("seg_target", "N/A")
VERSAO             = _comp_params.get("versao_ref", "N/A")
TREINO_EXEC_RUN_ID = _comp_params.get("treino_exec_run_id", "")

# ── n_df_model / n_df_validacao direto do PRE_PROC_MODEL exec run ─────────────
N_DF_MODEL     = None
N_DF_VALIDACAO = None

if PRE_PROC_MODEL_EXEC_RUN_ID:
    try:
        _pp_metrics = client.get_run(PRE_PROC_MODEL_EXEC_RUN_ID).data.metrics
        if "n_df_model" in _pp_metrics:
            N_DF_MODEL     = int(_pp_metrics["n_df_model"])
        if "n_df_validacao" in _pp_metrics:
            N_DF_VALIDACAO = int(_pp_metrics["n_df_validacao"])
    except Exception as _e:
        print(f"⚠️  Contagens não carregadas: {_e}")

print("• seg_target    :", SEG_TARGET)
print("• versão        :", VERSAO)
print("• n_df_model    :", N_DF_MODEL if N_DF_MODEL is not None else "N/A")
print("• n_df_validacao:", N_DF_VALIDACAO if N_DF_VALIDACAO is not None else "N/A")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loga params e métricas

# COMMAND ----------

mlflow.log_params({
    "pr_run_id":                  PR_RUN_ID,
    "pre_proc_exec_run_id":       PRE_PROC_EXEC_RUN_ID,
    "pre_proc_model_exec_run_id": PRE_PROC_MODEL_EXEC_RUN_ID,
    "fs_exec_run_id":             FS_EXEC_RUN_ID,
    "comp_exec_run_id":           COMP_EXEC_RUN_ID,
    "treino_exec_run_id":         TREINO_EXEC_RUN_ID,
    "seg_target":                 SEG_TARGET,
    "versao_ref":                 VERSAO,
    "mode_code":                  MODE_CODE,
    "report_versao":              REPORT_VERSAO,
})

if N_DF_MODEL is not None:
    mlflow.log_metric("n_df_model",     N_DF_MODEL)
if N_DF_VALIDACAO is not None:
    mlflow.log_metric("n_df_validacao", N_DF_VALIDACAO)

# metadata.json — resumo legível na raiz dos artefatos
_metadata = {
    "seg_target":                 SEG_TARGET,
    "versao_ref":                 VERSAO,
    "mode_code":                  MODE_CODE,
    "report_versao":              REPORT_VERSAO,
    "n_df_model":                 N_DF_MODEL,
    "n_df_validacao":             N_DF_VALIDACAO,
    "pre_proc_exec_run_id":       PRE_PROC_EXEC_RUN_ID,
    "pre_proc_model_exec_run_id": PRE_PROC_MODEL_EXEC_RUN_ID,
    "fs_exec_run_id":             FS_EXEC_RUN_ID,
    "comp_exec_run_id":           COMP_EXEC_RUN_ID,
    "treino_exec_run_id":         TREINO_EXEC_RUN_ID,
    "ts_exec":                    TS_EXEC,
}
with tempfile.TemporaryDirectory() as _td:
    _p = os.path.join(_td, "metadata.json")
    with open(_p, "w") as _f:
        json.dump(_metadata, _f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(_p)

print("✅ Params, métricas e metadata.json logados")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download e re-log de artefatos

# COMMAND ----------

TMPDIR = tempfile.mkdtemp()

# ── Artefatos do PRE_PROC run (1_PRE_PROC) → pre_proc/ ───────────────────────
_pre_proc_map = {
    "rules_execution.json": "pre_proc",
}
for _src_path, _dest_prefix in _pre_proc_map.items():
    try:
        _local = client.download_artifacts(PRE_PROC_EXEC_RUN_ID, _src_path, TMPDIR)
        mlflow.log_artifact(_local, artifact_path=_dest_prefix)
        print(f"✅ {_dest_prefix}/{os.path.basename(_src_path)}")
    except Exception as _e:
        print(f"⚠️  {_src_path} — {_e}")

# ── Artefatos do PRE_PROC_MODEL run (3_TREINO) → pre_proc_model/ ──────────────
_pre_proc_model_map = {
    "rules_execution.json":       "pre_proc_model",
    "profiling_df_model.json":    "pre_proc_model",
    "profiling_df_validacao.json": "pre_proc_model",
}
for _src_path, _dest_prefix in _pre_proc_model_map.items():
    try:
        _local = client.download_artifacts(PRE_PROC_MODEL_EXEC_RUN_ID, _src_path, TMPDIR)
        mlflow.log_artifact(_local, artifact_path=_dest_prefix)
        print(f"✅ {_dest_prefix}/{os.path.basename(_src_path)}")
    except Exception as _e:
        print(f"⚠️  {_src_path} — {_e}")

# ── Artefatos do FS run → feature_selection/ ──────────────────────────────────
_fs_map = {
    "pearson/pearson_heatmap.png":   "feature_selection",
    "summary/features_ranked.json":  "feature_selection",
}
for _src_path, _dest_prefix in _fs_map.items():
    try:
        _local = client.download_artifacts(FS_EXEC_RUN_ID, _src_path, TMPDIR)
        mlflow.log_artifact(_local, artifact_path=_dest_prefix)
        print(f"✅ {_dest_prefix}/{os.path.basename(_src_path)}")
    except Exception as _e:
        print(f"⚠️  {_src_path} — {_e}")

# ── Artefatos do COMP run → model_comparison/ ─────────────────────────────────
_comp_map = {
    "classification/pr_curves.png":              "model_comparison",
    "ranking/curves_ranking.png":                "model_comparison",
    "scores/score_distribution.png":             "model_comparison",
    "overfitting/overfitting_comparison.png":    "model_comparison",
    "concordance/rank_correlation_heatmap.png":  "model_comparison",
}
for _src_path, _dest_prefix in _comp_map.items():
    try:
        _local = client.download_artifacts(COMP_EXEC_RUN_ID, _src_path, TMPDIR)
        mlflow.log_artifact(_local, artifact_path=_dest_prefix)
        print(f"✅ {_dest_prefix}/{os.path.basename(_src_path)}")
    except Exception as _e:
        print(f"⚠️  {_src_path} — {_e}")

shutil.rmtree(TMPDIR, ignore_errors=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encerramento

# COMMAND ----------

while mlflow.active_run() is not None:
    mlflow.end_run()

print("✅ Runs encerradas")
print(f"\n• exec run        : {RUN_REPORT_EXEC}")
print(f"• seg_target      : {SEG_TARGET}")
print(f"• versão          : {VERSAO}")
print(f"\n⚠️  Artefatos logados na run '{RUN_REPORT_EXEC}':")
print("    metadata.json")
print("    pre_proc/rules_execution.json")
print("    pre_proc_model/rules_execution.json")
print("    pre_proc_model/profiling_df_model.json")
print("    pre_proc_model/profiling_df_validacao.json")
print("    feature_selection/pearson_heatmap.png")
print("    feature_selection/features_ranked.json")
print("    model_comparison/pr_curves.png")
print("    model_comparison/curves_ranking.png")
print("    model_comparison/score_distribution.png")
print("    model_comparison/overfitting_comparison.png")
print("    model_comparison/rank_correlation_heatmap.png")
