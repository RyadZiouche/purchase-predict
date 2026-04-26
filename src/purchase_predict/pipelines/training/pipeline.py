from kedro.pipeline import Pipeline, node, pipeline
from .nodes import auto_ml


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=auto_ml,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_test",
                    "y_test",
                    "params:automl_max_evals",
                    "params:mlflow_enabled",  # Nouvel interrupteur
                    "params:mlflow_experiment_id",  # Nouvel ID d'expérience
                ],
                outputs=dict(
                    model="model",
                    mlflow_run_id="mlflow_run_id",  # Nouvel output
                    mlflow_model_uri="mlflow_model_uri",  # Nouvel output
                ),
                name="auto_ml_node",
            )
        ]
    )
