from kedro.pipeline import Node, Pipeline
from .nodes import auto_ml


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=auto_ml,
                inputs=["X_train", "y_train", "X_test", "y_test", "params:automl_max_evals"],
                outputs=dict(model="model"),
                name="train_model_node",
            )
        ]
    )
