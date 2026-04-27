from kedro.pipeline import Pipeline, node
from .nodes import encode_features, split_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=encode_features,
                inputs="primary",
                outputs=dict(features="dataset", transform_pipeline="transform_pipeline"),
                name="encode_features_node",
            ),
            node(
                func=split_dataset,
                inputs=["dataset", "params:test_ratio"],
                outputs=dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_test="X_test",
                    y_test="y_test",
                ),
                name="split_dataset_node",
            ),
        ]
    )
