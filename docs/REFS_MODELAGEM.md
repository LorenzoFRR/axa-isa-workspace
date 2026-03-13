Referências modelagem de sistemas PROPENSITY_SCORE, LEAD_SCORING:

# Databricks Propensity Scoring – Solution Accelerator:
- https://www.databricks.com/solutions/accelerators/propensity-scoring?utm_source=chatgpt.com
- https://github.com/databricks-industry-solutions/propensity?utm_source=chatgpt.com
- Contexto de casos “propensity scoring mais complexos” (Feature Store, consistência treino↔inferência
    - https://www.databricks.com/blog/managing-complex-propensity-scoring-scenarios-databricks?utm_source=chatgpt.com

# Modelagem por probabilidade (Learning to rank):
## Se a forma como o time usa o modelo for “todo dia eu seleciono Top-K dentro de um contexto (ex.: por dia, por corretor, por carteira, por canal)”, você pode modelar como Learn-to-Rank com grupos (“qid”) e otimizar métricas como NDCG diretamente.
- Tutorial oficial do XGBoost sobre Learning to Rank (qid, rank:ndcg, LambdaMART
    - https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html?utm_source=chatgpt.com
- Post (PT-BR) explicando LTR no XGBoost
    - https://mariofilho.com/xgboost-learning-to-rank/?utm_source=chatgpt.com
- Exemplo prático em notebook (Kaggle / XGB ranker)
    - https://www.kaggle.com/code/radek1/training-an-xgboost-ranker-on-the-gpu?utm_source=chatgpt.com
- Docs do LightGBM para ranking (LGBMRanker)
    - https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html?utm_source=chatgpt.com


# Métricas de avaliação:
## Lift, Gains, Decis
    - explicação mais “de produto/negócio”: Lift chart em Microsoft Learn
        - https://learn.microsoft.com/en-us/analysis-services/data-mining/lift-chart-analysis-services-data-mining?view=asallproducts-allversions&utm_source=chatgpt.com
    - uso em contexto de analytics (Salesforce)
        - https://help.salesforce.com/s/articleView?id=analytics.bi_edd_model_metrics_cumulative_capture_category.htm&language=en_US&type=5&utm_source=chatgpt.com
    - referência complementar (seu uso de decil/quintil)
        - https://www.dasca.org/world-of-data-science/article/the-complete-guide-to-model-evaluation-metrics-in-data-science?utm_source=chatgpt.com
    - Exemplo de projeto com métricas de ranking aplicadas (ideia de storytelling e validação)
        - https://github.com/brunodifranco/project-insuricare-ranking?utm_source=chatgpt.com

# Regras de split / leakage
- TimeSeriesSplit (doc oficial do scikit-learn)
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com
- Guia de validação e cross-validation (sklearn)
    - https://scikit-learn.org/stable/modules/cross_validation.html?utm_source=chatgpt.com

# Calibração de probabilidade - interpretação Score
- Módulo de calibração do sklearn (conceitos)
    - https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com
- CalibratedClassifierCV (sigmoid/isotonic)
    - https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html?utm_source=chatgpt.com


# Importância de variáveis e explicabilidade
## se você tem segmentações (SEGURO_NOVO_MANUAL/DIGITAL etc.), evite comparar importâncias “na unha” sem controlar por drift de período e mix de dados; compare por janelas temporais consistentes e amostras equivalentes.
- Importância nativa do GBT no Apache Spark / PySpark: featureImportances
    - https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.GBTClassificationModel.html?utm_source=chatgpt.com
- SHAP (explicação mais confiável para árvores, incluindo TreeExplainer)
    - shap.TreeExplainer docs oficiais
        - https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html?utm_source=chatgpt.com
    - repo oficial SHAP (exemplos e padrões)
        - https://github.com/shap/shap?utm_source=chatgpt.com


# MLOps / Reprodutibilidade / Lineage
- Tracking (o que faz sentido logar: params/metrics/artifacts/code)
    - https://mlflow.org/docs/latest/ml/tracking/?utm_source=chatgpt.com
- Model Registry e lineage (run → modelo → versão)
    - https://mlflow.org/docs/latest/ml/model-registry/?utm_source=chatgpt.com
- Best practices de MLOps na Databricks (visão geral)
    - https://www.databricks.com/blog/mlops-best-practices-mlops-gym-crawl?utm_source=chatgpt.com