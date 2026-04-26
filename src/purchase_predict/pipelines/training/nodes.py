import numpy as np
import pandas as pd
from collections.abc import Callable
from typing import Any, TypedDict

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from lightgbm.sklearn import LGBMClassifier
from hyperopt import hp, tpe, fmin

import warnings
import os
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

warnings.filterwarnings("ignore")


# Définition du type pour la configuration des modèles
class ModelSpec(TypedDict, total=True):
    name: str
    model_class: type  # Utilisation de type pour la classe LGBMClassifier
    params: dict[str, Any]
    override_schemas: dict[str, type]


# Configuration du modèle LightGBM et de son espace de recherche
MODELS: list[ModelSpec] = [
    {
        "name": "LightGBM",
        "model_class": LGBMClassifier,
        "params": {
            "objective": "binary",
            "verbose": -1,
            "learning_rate": hp.uniform("learning_rate", 0.001, 1),
            "num_iterations": hp.quniform("num_iterations", 100, 1000, 20),
            "max_depth": hp.quniform("max_depth", 4, 12, 6),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 10),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "num_iterations": int,
        },
    }
]


def get_model_config(instance: BaseEstimator) -> ModelSpec:
    """Returns the configuration dictionary for the given model instance."""
    for model_spec in MODELS:
        model_cls: type = model_spec["model_class"]
        if isinstance(model_cls, type) and isinstance(instance, model_cls):
            return model_spec
    raise ValueError(f"Unsupported model: {type(instance)}")


def train_model(
    instance: BaseEstimator,
    training_set: tuple[np.ndarray, np.ndarray],
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Trains a new instance of model with supplied training set and hyper-parameters."""
    model_conf = get_model_config(instance)
    params = params or {}

    override_schemas = model_conf.get("override_schemas", {})
    for p in params:
        if p in override_schemas:
            params[p] = override_schemas[p](params[p])

    model = clone(instance)
    model.set_params(**params)
    model.fit(*training_set)
    return model


def optimize_hyp(
    instance: BaseEstimator,
    dataset: tuple[pd.DataFrame, pd.Series],
    search_space: dict,
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> dict:
    """Trains model's instances on hyper-parameters search space and returns most accurate."""
    X, y = dataset

    def objective(params):
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X.iloc[train_I, :]
            y_fold_train = y.iloc[train_I].values.flatten()
            X_fold_test = X.iloc[test_I, :]
            y_fold_test = y.iloc[test_I].values.flatten()

            model = train_model(
                instance=instance,
                training_set=(X_fold_train, y_fold_train),
                params=params,
            )
            scores_test.append(metric(y_fold_test, model.predict(X_fold_test)))

        return np.mean(scores_test)

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)


def save_pr_curve(X, y, model):
    os.makedirs(os.path.expanduser("data/08_reporting"), exist_ok=True)
    plt.figure(figsize=(16, 11))
    prec, recall, _ = precision_recall_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
    plt.title("PR Curve", fontsize=16)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig(os.path.expanduser("data/08_reporting/pr_curve.png"))
    plt.close()


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40,
    log_to_mlflow: bool = False,
    experiment_id: int = -1,
) -> dict[str, BaseEstimator | str]:
    """Runs training of multiple model instances and selects the most accurate."""

    # 1. Préparation sécurisée des données (Adaptation pour le parseur ty/astral du TP)
    X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)

    y_train_flat = y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    y_test_flat = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test
    y = pd.concat([pd.Series(y_train_flat), pd.Series(y_test_flat)], ignore_index=True)

    # 2. Entraînement et optimisation (Ton code d'origine)
    opt_models = []
    for model_specs in MODELS:
        optimum_params = optimize_hyp(
            model_specs["model_class"](),
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda x, y: -f1_score(x, y),
            max_evals=max_evals,
        )
        print(f"✅ Optimisation terminée pour : {model_specs['name']}")

        model = train_model(
            model_specs["model_class"](),
            training_set=(X_train, y_train),
            params=optimum_params,
        )
        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "score": f1_score(y_test, model.predict(X_test)),
            }
        )

    # Sélection du meilleur modèle
    best_model = max(opt_models, key=lambda x: x["score"])

    # ---------------------------------------------------------
    # 3. INTÉGRATION MLFLOW
    # ---------------------------------------------------------
    run_id = ""
    mlflow_model_uri = ""

    if log_to_mlflow:
        # Configuration du serveur
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER", "http://localhost:5000"))

        # Gestion de l'ID d'expérience
        exp_id = str(experiment_id)
        try:
            mlflow.get_experiment(exp_id)
        except mlflow.exceptions.MlflowException:
            if experiment_id == 1:
                exp_id = mlflow.create_experiment("purchase_predict")

        # Lancement du Tracking
        with mlflow.start_run(experiment_id=exp_id) as run:
            run_id = run.info.run_id

            # Enregistrement Paramètres & Score
            mlflow.log_params(best_model["params"])
            mlflow.log_metric("f1", best_model["score"])

            # Création et envoi du graphique PR Curve
            save_pr_curve(X_test, y_test, best_model["model"])
            mlflow.log_artifact("data/08_reporting/pr_curve.png", artifact_path="plots")

            # Signature et envoi du Modèle (.pkl)
            signature = infer_signature(X_train, best_model["model"].predict(X_train))
            mlflow_info = mlflow.sklearn.log_model(best_model["model"], "model", signature=signature)
            mlflow_model_uri = mlflow_info.model_uri

    # 4. On retourne le dictionnaire enrichi à Kedro
    return {
        "model": best_model["model"],
        "mlflow_run_id": run_id,
        "mlflow_model_uri": mlflow_model_uri,
    }
