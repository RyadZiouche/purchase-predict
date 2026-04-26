import pytest
from purchase_predict.pipelines.loading.nodes import load_csv_from_bucket
from purchase_predict.pipelines.processing.nodes import encode_features
from kedro.io import DataCatalog, MemoryDataset


@pytest.fixture(scope="module")
def project_id():
    # ⚠️ REMPLACE PAR TON VRAI PROJECT ID GCP
    return "purchase-predict-project"


@pytest.fixture(scope="module")
def primary_folder():
    # ⚠️ REMPLACE PAR TON VRAI CHEMIN VERS LE FICHIER CSV
    return "purchase-predict-ryad/primary.csv"


@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, primary_folder):
    # Cette fixture télécharge les données brutes avant qu'elles ne soient modifiées
    return load_csv_from_bucket(project_id, primary_folder)


@pytest.fixture(scope="module")
def test_ratio():
    return 0.3


@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    # On simule le passage dans le premier nœud pour avoir des données encodées
    return encode_features(dataset_not_encoded)["features"]


@pytest.fixture(scope="module")
def catalog_test(dataset_not_encoded, test_ratio):
    catalog = DataCatalog(
        {"primary": MemoryDataset(dataset_not_encoded), "params:test_ratio": MemoryDataset(test_ratio)}
    )
    return catalog
