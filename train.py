"""
train.py
Projeto: Ferramentas de Monitoramento de Bem-Estar e Saúde Mental no Trabalho
Integrantes: Diogo Julio - RM553837; Jonata Rafael - RM552939
Instituição: FIAP – Engenharia de Software 3ESPR

O script gera um dataset sintético representativo e executa:
1) Classificação de risco de burnout (Logistic Regression, RandomForest)
2) Regressão para prever humor diário (Regressão Linear, RandomForest Regressor)
3) Clusterização de perfis (KMeans)

Produz gráficos e um relatório PDF em outputs/
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             mean_squared_error, r2_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans
from fpdf import FPDF
import joblib

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "best_models").mkdir(exist_ok=True)

def generate_synthetic_dataset(n=1000, random_state=42):
    np.random.seed(random_state)
    # Continuous features
    hours_sleep = np.clip(np.random.normal(7, 1.2, n), 3, 10)             # horas de sono
    hours_work = np.clip(np.random.normal(8, 2.0, n), 2, 14)             # horas trabalhadas
    breaks = np.clip(np.random.poisson(2, n), 0, 10)                     # número de pausas
    meetings = np.clip(np.random.poisson(3, n), 0, 12)                   # reuniões por dia
    commute = np.clip(np.random.normal(30, 20, n), 0, 180)               # tempo de deslocamento (min)
    exercise = np.clip(np.random.beta(2,5, n)*7, 0, 7)                   # dias/semana com exercício (~0-7)
    social = np.clip(np.random.normal(3,1.5,n), 0, 10)                   # interação social (0-10)
    caffeine = np.clip(np.random.poisson(2, n), 0, 8)                    # cafés por dia
    screen_time = np.clip(np.random.normal(6,2,n), 0, 16)                # horas de tela fora do trabalho
    job_satisfaction = np.clip(np.random.beta(2,2,n)*10, 0, 10)          # 0-10
    perceived_stress = np.clip( (10 - job_satisfaction) + np.random.normal(0,2,n), 0, 10)
    # Derived / target variables
    # mood: 0-10 continuous (higher = melhor humor)
    mood = np.clip( (hours_sleep/8)*2 + (job_satisfaction/10)*4 + (exercise/7)*1.5 + (social/10)*1.0 - (perceived_stress/10)*3 + np.random.normal(0,1,n), 0, 10)

    # burnout risk label (binary) - 1 = at risk
    prob_burnout = ( (hours_work - 8)/6 + (5 - hours_sleep)/5 + (3 - breaks)/3 + (perceived_stress/10) + (5 - job_satisfaction)/5 )
    prob_burnout = 1 / (1 + np.exp(-prob_burnout))  # sigmoid to map to 0-1
    burnout_risk = (prob_burnout + np.random.normal(0,0.1,n)) > 0.6
    burnout_risk = burnout_risk.astype(int)

    df = pd.DataFrame({
        "hours_sleep": hours_sleep,
        "hours_work": hours_work,
        "breaks": breaks,
        "meetings": meetings,
        "commute_min": commute,
        "exercise_days": exercise,
        "social_interaction": social,
        "coffee_cups": caffeine,
        "screen_time_outside_work": screen_time,
        "job_satisfaction": job_satisfaction,
        "perceived_stress": perceived_stress,
        "mood": mood,
        "burnout_risk": burnout_risk
    })
    return df

def classification_task(df):
    print("=== Classification task: Burnout risk ===")
    features = ["hours_sleep","hours_work","breaks","meetings","commute_min","exercise_days","social_interaction","coffee_cups","screen_time_outside_work","job_satisfaction","perceived_stress"]
    X = df[features]
    y = df["burnout_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_proba_lr = lr.predict_proba(X_test_s)[:,1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]

    # Metrics
    def metrics(y_true, y_pred, y_proba=None):
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
        if y_proba is not None:
            try:
                out["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                out["roc_auc"] = None
        return out

    lr_metrics = metrics(y_test, y_pred_lr, y_proba_lr)
    rf_metrics = metrics(y_test, y_pred_rf, y_proba_rf)
    print("LogisticRegression:", lr_metrics)
    print("RandomForest:", rf_metrics)

    # Save models and scaler
    joblib.dump(lr, OUTPUT_DIR / "best_models" / "clf_logistic.pkl")
    joblib.dump(rf, OUTPUT_DIR / "best_models" / "clf_rf.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "best_models" / "scaler_clf.pkl")

    # ROC Curve plot for LR
    fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'LogisticRegression (AUC={lr_metrics.get("roc_auc"):.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC - Burnout Risk (Logistic Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_burnout_lr.png")
    plt.close()

    # Confusion matrix for RF
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - RandomForest (Burnout)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cm_burnout_rf.png")
    plt.close()

    return {"lr_metrics": lr_metrics, "rf_metrics": rf_metrics}

def regression_task(df):
    print("=== Regression task: Predict mood ===")
    features = ["hours_sleep","hours_work","breaks","meetings","commute_min","exercise_days","social_interaction","coffee_cups","screen_time_outside_work","job_satisfaction","perceived_stress"]
    X = df[features]
    y = df["mood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Metrics
    def reg_metrics(y_true, y_pred):
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "r2": r2_score(y_true, y_pred)
        }

    lr_m = reg_metrics(y_test, y_pred_lr)
    rf_m = reg_metrics(y_test, y_pred_rf)
    print("LinearRegression:", lr_m)
    print("RandomForestRegressor:", rf_m)

    # Save models and scaler
    joblib.dump(lr, OUTPUT_DIR / "best_models" / "reg_linear.pkl")
    joblib.dump(rf, OUTPUT_DIR / "best_models" / "reg_rf.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "best_models" / "scaler_reg.pkl")

    # Scatter plot: true vs pred for RF
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.xlabel("True mood")
    plt.ylabel("Predicted mood (RF)")
    plt.title("True vs Predicted Mood")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reg_true_vs_pred_rf.png")
    plt.close()

    return {"linear": lr_m, "rf": rf_m}

def clustering_task(df, n_clusters=4):
    print("=== Clustering task: KMeans profiles ===")
    features = ["hours_sleep","hours_work","breaks","exercise_days","social_interaction","job_satisfaction","perceived_stress","mood"]
    X = df[features].copy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_s)
    df["cluster"] = labels

    # Save kmeans model and scaler
    joblib.dump(kmeans, OUTPUT_DIR / "best_models" / "kmeans.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "best_models" / "scaler_kmeans.pkl")

    # Plot clusters (use PCA for visualization)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_s)
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=comps[:,0], y=comps[:,1], hue=labels, palette="tab10", alpha=0.7)
    plt.title("KMeans clusters (2D projection)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_clusters.png")
    plt.close()

    # Cluster summary
    summary = df.groupby("cluster").agg({
        "hours_sleep":"mean","hours_work":"mean","breaks":"mean","exercise_days":"mean",
        "social_interaction":"mean","job_satisfaction":"mean","perceived_stress":"mean","mood":"mean","burnout_risk":"mean"
    }).round(3)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    return summary

def generate_report(classif_res, reg_res, cluster_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Relatorio - Ferramentas de Monitoramento de Bem-Estar e Saude Mental no Trabalho", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0,6,"Integrantes: Diogo Julio - RM553837; Jonata Rafael - RM552939")
    pdf.multi_cell(0,6,"Instituicao: FIAP – Engenharia de Software 3ESPR")
    pdf.ln(4)
    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0,6,"1) Classificacao - Burnout Risk", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,6, f"LogisticRegression: {classif_res['lr_metrics']}")
    pdf.multi_cell(0,6, f"RandomForest: {classif_res['rf_metrics']}")
    pdf.ln(3)

    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0,6,"2) Regressao - Predicao de Humor", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,6, f"LinearRegression: {reg_res['linear']}")
    pdf.multi_cell(0,6, f"RandomForestRegressor: {reg_res['rf']}")
    pdf.ln(3)

    pdf.set_font("Arial", style="B", size=11)
    pdf.cell(0,6,"3) Clusterizacao - Perfis de Bem-Estar", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0,6, "Cluster summary (saved as CSV in outputs/cluster_summary.csv)")
    pdf.ln(4)

    for img in ["roc_burnout_lr.png","cm_burnout_rf.png","reg_true_vs_pred_rf.png","kmeans_clusters.png"]:
        p = OUTPUT_DIR / img
        if p.exists():
            try:
                pdf.image(str(p), x=10, y=None, w=190)
                pdf.ln(4)
            except Exception:
                pass

    pdf.output(OUTPUT_DIR / "report.pdf")

def main():
    df = generate_synthetic_dataset(n=1200)
    df.to_csv(OUTPUT_DIR / "synthetic_wellbeing_dataset.csv", index=False)
    classif_res = classification_task(df)
    reg_res = regression_task(df)
    cluster_summary = clustering_task(df, n_clusters=4)
    generate_report(classif_res, reg_res, cluster_summary)
    print("Done. Outputs saved to outputs/")

if __name__ == "__main__":
    main()
