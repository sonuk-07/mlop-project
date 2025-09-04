import logging
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
from pipelines.modeling import evaluate_model
import os
import pickle

log = logging.getLogger(__name__)

def train_catboost_model(
    X_train, y_train, X_test, y_test, hyperparams=None, artifact_dir="/opt/airflow/artifacts"
):
    """
    Train a CatBoost model, evaluate it, log everything to MLflow,
    save both .cbm and .pkl formats for deployment, and return XCom-friendly info.
    """
    if any(v is None or len(v) == 0 for v in [X_train, y_train, X_test, y_test]):
        raise ValueError("Training or testing data is empty or None.")

    # Ensure artifact directory exists
    os.makedirs(artifact_dir, exist_ok=True)
    log.info("Starting CatBoost model training.")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Start MLflow run
    with mlflow.start_run(run_name="CatBoost_Run") as run:
        run_id = run.info.run_id
        log.info(f"MLflow run started with ID: {run_id}")

        # Use provided hyperparameters or defaults
        if hyperparams:
            log.info(f"Using tuned hyperparameters: {hyperparams}")
            model = CatBoostClassifier(**hyperparams, random_state=42, verbose=0)
            mlflow.log_params(hyperparams)
        else:
            default_params = {
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 5,
                "random_state": 42
            }
            log.info("No hyperparameters provided; using default parameters.")
            model = CatBoostClassifier(**default_params, verbose=0)
            mlflow.log_params(default_params)

        # Train the model
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100
        )

        log.info("Training complete. Evaluating model...")
        metrics = evaluate_model(
            "CatBoost", model, X_train, y_train, X_test, y_test, artifact_dir=artifact_dir
        )
        log.info(f"Evaluation metrics: {metrics}")

        # -----------------------
        # Save model locally
        # -----------------------
        model_cbm_path = os.path.join(artifact_dir, "catboost_model.cbm")
        model.save_model(model_cbm_path)

        model_pkl_path = os.path.join(artifact_dir, "catboost_model.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(model, f)

        # -----------------------
        # Log artifacts to MLflow
        # -----------------------
        mlflow.log_artifact(model_cbm_path, artifact_path="catboost_artifacts")
        mlflow.log_artifact(model_pkl_path, artifact_path="catboost_artifacts")


        # -----------------------
        # Log CatBoost model to MLflow Model Registry
        # -----------------------
        mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="catboost_model",
            registered_model_name="CatBoostClassifierModel"
        )

        # -----------------------
        # Log evaluation metrics to MLflow
        # -----------------------
        mlflow.log_metrics({
            "train_accuracy": metrics.get("train_accuracy"),
            "test_accuracy": metrics.get("test_accuracy"),
            "roc_auc": metrics.get("roc_auc")
        })

        log.info(f"Model saved locally and logged to MLflow. Run ID: {run_id}")

    # -----------------------
    # Return XCom-friendly info
    # -----------------------
    return {
        "model_cbm_path": model_cbm_path,  # local .cbm path
        "model_pkl_path": model_pkl_path,  # local .pkl path
        "mlflow_run_id": run_id,
        "mlflow_artifact_path": f"mlflow:catboost_artifacts",  # logical MLflow path
        "metrics": metrics
    }
