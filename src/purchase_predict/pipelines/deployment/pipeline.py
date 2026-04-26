from kedro.pipeline import Pipeline, node, pipeline
from .nodes import push_to_model_registry, stage_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=push_to_model_registry,
                inputs=["params:mlflow_model_registry", "mlflow_model_uri"],
                outputs="mlflow_model_version",
                name="push_to_registry_node",
            ),
            node(
                func=stage_model,
                inputs=["params:mlflow_model_registry", "mlflow_model_version"],
                outputs=None,
                name="stage_model_node",
            ),
        ]
    )
