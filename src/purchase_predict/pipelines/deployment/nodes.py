import os
import mlflow
from mlflow.tracking import MlflowClient


def push_to_model_registry(registry_name: str, model_uri: str) -> str:
    """Envoie une version du modèle vers le Model Registry."""
    tracking_uri = os.getenv("MLFLOW_SERVER")
    if not tracking_uri:
        raise ValueError("La variable MLFLOW_SERVER n'est pas définie dans .env")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Création d'une nouvelle version dans le registre
    result = client.create_model_version(
        name=registry_name,
        source=model_uri,
    )
    return result.version


def stage_model(registry_name: str, version: str) -> None:
    """Assigne l'alias 'staging' ou 'production' à la version du modèle."""
    env = os.getenv("ENV")
    if env is None:
        return

    client = MlflowClient()
    # Ajoute l'alias (ex: 'staging') au modèle
    client.set_registered_model_alias(name=registry_name, alias=env, version=version)
