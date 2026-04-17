import pandas as pd
from kedro.runner import SequentialRunner
from purchase_predict.pipelines.loading.pipeline import create_pipeline


def test_pipeline(catalog_test):
    # On crée un "coureur" qui va exécuter l'usine
    runner = SequentialRunner()
    # On crée l'usine
    pipeline = create_pipeline()

    # On lance l'usine avec notre faux catalogue de test !
    pipeline_output = runner.run(pipeline, catalog_test)

    # On récupère le résultat
    df = pipeline_output["primary"].load()

    # On fait les mêmes vérifications
    assert type(df) is pd.DataFrame
    assert df.shape[1] == 16
    assert "purchased" in df
