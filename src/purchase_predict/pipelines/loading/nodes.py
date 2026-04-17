import pandas as pd
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads a single CSV file from bucket.
    """
    storage_client = storage.Client()

    # Découpage pour trouver le nom du bucket et le nom du fichier
    bucket_name = bucket_path.split("/")[0]
    file_name = "/".join(bucket_path.split("/")[1:])

    # Connexion au bucket Google Cloud
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Téléchargement à la racine du projet (Compatible Windows)
    local_path = "primary_downloaded.csv"
    blob.download_to_filename(local_path)

    # Lecture du CSV avec Pandas
    df = pd.read_csv(local_path)
    return df
