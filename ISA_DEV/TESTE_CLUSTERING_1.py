# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

# =========================
# INPUT
# =========================
INPUT_TABLE   = "silver.cotacao_seg_YYYYMMDD_HHMMSS"   # <<< AJUSTE

# =========================
# FILTRO DE SEG (opcional)
# =========================
# None = todos os SEGs; ou ex: "SEGURO_NOVO_MANUAL"
SEG_FILTER    = None

# =========================
# CLUSTERING
# =========================
K_RANGE       = [3, 4, 5]   # Ks a avaliar no elbow/silhouette
K_FINAL       = 4            # K escolhido para o resultado final
RANDOM_SEED   = 42

# =========================
# NULOS nas features _detalhe
# =========================
# "drop"          → corretores com qualquer NULL são excluídos do clustering
# "impute_median" → NULLs substituídos pela mediana da feature
NULL_STRATEGY = "drop"

print("✅ Config carregada")
print("• input        :", INPUT_TABLE)
print("• seg_filter   :", SEG_FILTER)
print("• k_range      :", K_RANGE)
print("• k_final      :", K_FINAL)
print("• random_seed  :", RANDOM_SEED)
print("• null_strategy:", NULL_STRATEGY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pyspark.sql import functions as F

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load

# COMMAND ----------

df_raw = spark.table(INPUT_TABLE)

if SEG_FILTER is not None:
    df_raw = df_raw.filter(F.col("SEG") == SEG_FILTER)
    print(f"• Filtro SEG aplicado: {SEG_FILTER}")

df_raw = df_raw.select(
    "CD_DOC_CORRETOR",
    "DS_PRODUTO_NOME",
    "HR_2025_detalhe",
    "QTD_COTACAO_2025_detalhe",
)

print(f"• Linhas carregadas      : {df_raw.count():,}")
print(f"• Corretores distintos   : {df_raw.select('CD_DOC_CORRETOR').distinct().count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agregação — nível corretor

# COMMAND ----------

df_corretor = (
    df_raw.groupBy("CD_DOC_CORRETOR")
    .agg(
        F.mean("HR_2025_detalhe").cast("double").alias("hr_mean"),
        F.mean("QTD_COTACAO_2025_detalhe").cast("double").alias("cotacao_mean"),
        F.countDistinct("DS_PRODUTO_NOME").alias("n_produtos"),
    )
)

total_corretores = df_corretor.count()
print(f"• Corretores únicos: {total_corretores:,}")

# Profiling de NULLs
for col in ["hr_mean", "cotacao_mean", "n_produtos"]:
    n_null = df_corretor.filter(F.col(col).isNull()).count()
    print(f"  NULLs em {col}: {n_null} ({n_null/total_corretores*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento

# COMMAND ----------

CLUSTER_FEATURES = ["hr_mean", "cotacao_mean", "n_produtos"]

pdf = df_corretor.toPandas()

# NULL handling
if NULL_STRATEGY == "drop":
    n_antes = len(pdf)
    pdf = pdf.dropna(subset=CLUSTER_FEATURES)
    print(f"• Corretores removidos por NULL: {n_antes - len(pdf):,} → {len(pdf):,} corretores restantes")
elif NULL_STRATEGY == "impute_median":
    for col in CLUSTER_FEATURES:
        med = pdf[col].median()
        n_null = pdf[col].isna().sum()
        pdf[col] = pdf[col].fillna(med)
        print(f"  Imputação {col}: {n_null} NULLs → mediana {med:.4f}")

# Escalar (obrigatório para K-Means)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(pdf[CLUSTER_FEATURES])

print(f"• Shape matriz clustering: {X_scaled.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seleção de K — Elbow + Silhouette

# COMMAND ----------

inertias    = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"  K={k} | inertia={km.inertia_:,.1f} | silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_RANGE, inertias, marker="o")
axes[0].set_title("Elbow — Inertia por K")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")
axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

axes[1].plot(K_RANGE, silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score por K")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Silhouette")
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit final — K_FINAL

# COMMAND ----------

km_final = KMeans(n_clusters=K_FINAL, random_state=RANDOM_SEED, n_init=10)
pdf["CLF_CORRETOR"] = km_final.fit_predict(X_scaled)

print(f"• K={K_FINAL} | inertia={km_final.inertia_:,.1f}")
print(f"• Silhouette: {silhouette_score(X_scaled, pdf['CLF_CORRETOR']):.4f}")
print()
print("Corretores por cluster:")
print(pdf["CLF_CORRETOR"].value_counts().sort_index().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análise dos clusters

# COMMAND ----------

# ── Perfil médio por cluster ──────────────────────────────────────────────────
cluster_profile = (
    pdf.groupby("CLF_CORRETOR")[CLUSTER_FEATURES]
    .agg(["mean", "median", "std", "count"])
)
print("Perfil por cluster (mean / median / std / n):")
display(cluster_profile)

# ── Heatmap de centroides (escala original) ───────────────────────────────────
centroids_orig = scaler.inverse_transform(km_final.cluster_centers_)
df_centroids   = pd.DataFrame(centroids_orig, columns=CLUSTER_FEATURES)
df_centroids.index.name = "CLF_CORRETOR"

fig, ax = plt.subplots(figsize=(7, max(3, K_FINAL)))
im = ax.imshow(
    (df_centroids - df_centroids.mean()) / df_centroids.std(),
    aspect="auto", cmap="RdYlGn"
)
ax.set_xticks(range(len(CLUSTER_FEATURES)))
ax.set_xticklabels(CLUSTER_FEATURES, rotation=25, ha="right")
ax.set_yticks(range(K_FINAL))
ax.set_yticklabels([f"Cluster {i}" for i in range(K_FINAL)])
ax.set_title("Centroides normalizados por feature")
plt.colorbar(im, ax=ax, label="z-score (relativo à média global)")
for i in range(K_FINAL):
    for j, col in enumerate(CLUSTER_FEATURES):
        ax.text(j, i, f"{df_centroids.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="black")
plt.tight_layout()
plt.show()

# ── Scatter: HR vs QTD_COTACAO ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = plt.cm.tab10.colors

for cluster_id in sorted(pdf["CLF_CORRETOR"].unique()):
    sub = pdf[pdf["CLF_CORRETOR"] == cluster_id]
    axes[0].scatter(sub["cotacao_mean"], sub["hr_mean"],
                    label=f"Cluster {cluster_id}", alpha=0.5, s=20,
                    color=colors[cluster_id % len(colors)])
    axes[1].scatter(sub["n_produtos"], sub["hr_mean"],
                    label=f"Cluster {cluster_id}", alpha=0.5, s=20,
                    color=colors[cluster_id % len(colors)])

axes[0].set_xlabel("cotacao_mean")
axes[0].set_ylabel("hr_mean")
axes[0].set_title("HR vs Volume de cotações")
axes[0].legend()

axes[1].set_xlabel("n_produtos")
axes[1].set_ylabel("hr_mean")
axes[1].set_title("HR vs Diversidade de produtos")
axes[1].legend()

plt.tight_layout()
plt.show()

# ── Corretores representativos (mais próximos ao centroide) ───────────────────
dists      = cdist(X_scaled, km_final.cluster_centers_)
pdf["dist_centroide"] = dists[range(len(pdf)), pdf["CLF_CORRETOR"]]

print("\nCorretores mais próximos ao centroide por cluster:")
rep = (
    pdf.sort_values("dist_centroide")
    .groupby("CLF_CORRETOR")
    .head(3)
    [["CD_DOC_CORRETOR", "CLF_CORRETOR", "hr_mean", "cotacao_mean", "n_produtos", "dist_centroide"]]
)
display(rep.sort_values(["CLF_CORRETOR", "dist_centroide"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output — df_resultado

# COMMAND ----------

df_resultado = pdf[["CD_DOC_CORRETOR", "CLF_CORRETOR"]].copy()
df_resultado["CLF_CORRETOR"] = df_resultado["CLF_CORRETOR"].astype(int)

print(f"• Total corretores classificados: {len(df_resultado):,}")
print()
display(df_resultado.sort_values("CLF_CORRETOR"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Opcional) Join de volta à tabela de cotações
# MAGIC
# MAGIC Para validar como CLF_CORRETOR ficaria distribuído nas cotações:
# MAGIC ```python
# MAGIC df_resultado_sp = spark.createDataFrame(df_resultado)
# MAGIC df_com_clf = df_raw.join(df_resultado_sp, on="CD_DOC_CORRETOR", how="left")
# MAGIC display(df_com_clf.groupBy("CLF_CORRETOR").count().orderBy("CLF_CORRETOR"))
# MAGIC ```
