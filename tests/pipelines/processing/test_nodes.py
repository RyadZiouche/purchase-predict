import pandas as pd
from purchase_predict.pipelines.processing.nodes import encode_features
import numpy as np
from purchase_predict.pipelines.processing.nodes import split_dataset

# Nos règles métiers très strictes
BALANCE_THRESHOLD = 0.1  # Au moins 10% d'acheteurs
MIN_SAMPLES = 5000  # Au moins 5000 lignes de données


def test_encode_features(dataset_not_encoded):
    # On fait passer nos données dans la machine de nettoyage
    encoded = encode_features(dataset_not_encoded)
    df = encoded["features"]

    # 1. Vérifier que c'est bien un tableau de données
    assert isinstance(df, pd.DataFrame), "Expected DataFrame for features"

    # 2. Vérifier que la colonne cible ne contient QUE des 0 et des 1
    assert df["purchased"].isin([0, 1]).all()

    # 3. Vérifier que TOUTES les colonnes sont devenues des nombres (pas de texte !)
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df.dtypes[col])

    # 4. Vérifier qu'on a assez de données pour entraîner un modèle
    assert df.shape[0] > MIN_SAMPLES

    # 5. Vérifier que la donnée n'est pas trop déséquilibrée
    assert (df["purchased"].value_counts() / df.shape[0] > BALANCE_THRESHOLD).all()


def test_split_dataset(dataset_encoded, test_ratio):
    # On fait passer les données dans notre fonction de découpage
    split_result = split_dataset(dataset_encoded, test_ratio)
    X_train, y_train, X_test, y_test = split_result.values()

    # 1. Vérifie qu'on a autant de réponses (y) que de questions (X)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # 2. Vérifie qu'aucune ligne n'a disparu lors du découpage (Train + Test = Total)
    assert X_train.shape[0] + X_test.shape[0] == dataset_encoded.shape[0]

    # 3. Vérifie que le ratio de 30% a bien été respecté
    assert np.ceil(dataset_encoded.shape[0] * test_ratio) == X_test.shape[0]
