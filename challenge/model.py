import pandas as pd

from typing import Tuple, Union, List


class ReplenishmentModel:

    def __init__(
        self
    ):
        self._model = None  # El modelo debe guardarse en este atributo.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos crudos para entrenamiento o predicción.

        Args:
            data (pd.DataFrame): datos crudos.
            target_column (str, opcional): si se establece, se retorna el target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features y target.
            o
            pd.DataFrame: features.
        """
        return

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Entrena el modelo con los datos preprocesados.

        Args:
            features (pd.DataFrame): datos preprocesados.
            target (pd.DataFrame): variable objetivo.
        """
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[dict]:
        """
        Predice el consumo para una lista de productos.

        Args:
            features (pd.DataFrame): datos preprocesados.

        Returns:
            (List[dict]): predicciones con keys 'fecha' y 'cantidad'.
        """
        return

    def save(
        self,
        path: str
    ) -> None:
        """
        Guarda el modelo entrenado en disco.

        Args:
            path (str): ruta donde guardar el modelo.
        """
        return

    def load(
        self,
        path: str
    ) -> None:
        """
        Carga un modelo entrenado desde disco.

        Args:
            path (str): ruta desde donde cargar el modelo.
        """
        return