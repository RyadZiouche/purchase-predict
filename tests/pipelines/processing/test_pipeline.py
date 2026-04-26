from kedro.runner import SequentialRunner
from purchase_predict.pipelines.processing.pipeline import create_pipeline


def test_pipeline(catalog_test):
    # 1. On prépare le moteur (Runner) et l'usine (Pipeline)
    runner = SequentialRunner()
    pipeline = create_pipeline()

    # 2. On lance l'usine avec notre faux catalogue !
    pipeline_output = runner.run(pipeline, catalog_test)

    # 3. On récupère ce qui sort à la fin de la chaîne
    X_train = pipeline_output["X_train"].load()
    y_train = pipeline_output["y_train"].load()
    X_test = pipeline_output["X_test"].load()
    y_test = pipeline_output["y_test"].load()

    # 4. On fait une vérification rapide
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
