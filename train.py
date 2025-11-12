"""
train.py
Treina modelos no dataset Breast Cancer e gera relatório PDF com resultados.
Execução: python train.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report)
import matplotlib.pyplot as plt
from fpdf import FPDF
import joblib

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]
    feature_names = X.columns.tolist()
    return X, y, feature_names

def evaluate_model(clf, X, y, cv):
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    summary = {k: float(np.mean(v)) for k, v in scores.items()}
    return summary, scores

def main():
    X, y, feature_names = load_data()
    # train/test split for final evaluation and ROC plot
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Pipeline for Logistic Regression (with regularization)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
    ])
    param_grid_lr = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"]
    }
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs_lr.fit(X_train, y_train)

    # Random Forest baseline
    pipe_rf = Pipeline([
        ("scaler", StandardScaler()),  # not strictly necessary for RF, but kept for consistency
        ("clf", RandomForestClassifier(random_state=42))
    ])
    param_grid_rf = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 5, 10]
    }
    gs_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs_rf.fit(X_train, y_train)

    # Evaluate best models with cross-validation
    best_lr = gs_lr.best_estimator_
    best_rf = gs_rf.best_estimator_

    lr_summary, lr_scores = evaluate_model(best_lr, X, y, cv)
    rf_summary, rf_scores = evaluate_model(best_rf, X, y, cv)

    # Final test set evaluation and ROC curves
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:,1]
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:,1]

    metrics = {
        "LogisticRegression": {
            "accuracy": accuracy_score(y_test, y_pred_lr),
            "precision": precision_score(y_test, y_pred_lr),
            "recall": recall_score(y_test, y_pred_lr),
            "f1": f1_score(y_test, y_pred_lr),
            "roc_auc": roc_auc_score(y_test, y_proba_lr)
        },
        "RandomForest": {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "precision": precision_score(y_test, y_pred_rf),
            "recall": recall_score(y_test, y_pred_rf),
            "f1": f1_score(y_test, y_pred_rf),
            "roc_auc": roc_auc_score(y_test, y_proba_rf)
        }
    }

    # Save best model (the one with higher roc_auc on test)
    chosen_model_name = "LogisticRegression" if metrics["LogisticRegression"]["roc_auc"] >= metrics["RandomForest"]["roc_auc"] else "RandomForest"
    best_model = best_lr if chosen_model_name == "LogisticRegression" else best_rf
    joblib.dump(best_model, OUTPUT_DIR / "best_model.pkl")

    # Create ROC plot
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f'LogisticRegression (AUC = {metrics["LogisticRegression"]["roc_auc"]:.3f})')
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {metrics["RandomForest"]["roc_auc"]:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test set")
    plt.legend(loc="lower right")
    roc_path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # Generate classification reports
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    # Write a PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10,"Relatorio - Classificacao Breast Cancer", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Dataset: Breast Cancer Wisconsin (sklearn.datasets)\nNumero de instancias: {X.shape[0]}\nNumero de features: {X.shape[1]}")
    pdf.ln(4)
    pdf.multi_cell(0,6, "Modelos avaliados: LogisticRegression (regularizacao L2) e RandomForest (baseline).")
    pdf.ln(6)
    pdf.cell(0,6,"Resultados (metricas no conjunto de teste):", ln=True)
    pdf.ln(2)
    for mname, m in metrics.items():
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0,6, mname, ln=True)
        pdf.set_font("Arial", size=11)
        for k,v in m.items():
            pdf.cell(0,6, f"  - {k}: {v:.4f}", ln=True)
        pdf.ln(2)
    # Insert ROC image if exists
    if roc_path.exists():
        pdf.image(str(roc_path), x=10, y=None, w=190)
    pdf.output(str(OUTPUT_DIR / "report.pdf"))
    print("Training script completed. Outputs saved to outputs/: best_model.pkl, report.pdf, roc_curve.png")
    print(f"Chosen model based on test ROC AUC: {chosen_model_name}")

if __name__ == "__main__":
    main()
