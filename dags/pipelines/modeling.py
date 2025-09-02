import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
import optuna
import mlflow
import mlflow.catboost
import os

log = logging.getLogger(__name__)

def evaluate_model(name, model, X_train, y_train, X_test, y_test, artifact_dir="/opt/airflow/artifacts"):
    """
    Evaluates a trained model, logs metrics and artifacts to MLflow,
    and returns key metrics as a dictionary (XCom-safe).
    """
    os.makedirs(artifact_dir, exist_ok=True)
    log.info(f"Starting evaluation for {name}...")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_test = model.decision_function(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    report = classification_report(y_test, y_pred_test)
    log.info(f"\n{name} Classification Report:\n{report}")

    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    # Save artifacts
    report_path = os.path.join(artifact_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Confusion Matrix
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ROC Curve
    roc_path = os.path.join(artifact_dir, "roc_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("roc_auc", roc_auc)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path)

    # Precision-Recall Curve
    pr_path = os.path.join(artifact_dir, "precision_recall_curve.png")
    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{name}")
    plt.title(f"{name} - Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(pr_path)
    plt.close()
    mlflow.log_artifact(pr_path)

    log.info(f"Evaluation complete for {name}.")
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "roc_auc": roc_auc
    }


def tune_catboost(X_train, y_train, n_trials=3):
    """
    Hyperparameter tuning for CatBoost using Optuna.
    Logs metrics and parameters to MLflow.
    Returns best parameters dict (XCom-safe).
    """
    if X_train is None or X_train.empty or y_train is None or y_train.empty:
        raise ValueError("Training data is empty or None.")

    log.info(f"Starting Optuna tuning for {n_trials} trials.")

    def objective(trial):
        with mlflow.start_run(nested=True):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 5)
            }
            mlflow.log_params(params)
            model = CatBoostClassifier(**params, random_state=42, verbose=0)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1).mean()
            mlflow.log_metric("f1_score", score)
            return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    log.info(f"Best hyperparameters: {study.best_params}")
    
    return study.best_params  # Safe to push to Airflow XCom
