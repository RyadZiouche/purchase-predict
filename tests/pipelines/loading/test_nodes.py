import pandas as pd
from purchase_predict.pipelines.loading.nodes import load_csv_from_bucket


def test_load_csv_from_bucket(project_id, primary_folder):
    # On exécute la fonction avec les fixtures
    df = load_csv_from_bucket(project_id, primary_folder)

    # 1. On vérifie que c'est bien un DataFrame Pandas (tableau)
    assert type(df) is pd.DataFrame
    # 2. On vérifie qu'il y a bien 16 colonnes
    assert df.shape[1] == 16
    # 3. On vérifie que la colonne cible "purchased" existe bien
    assert "purchased" in df
