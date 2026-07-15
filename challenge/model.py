import os
import pickle

import numpy as np
import pandas as pd

from typing import Tuple, Union, List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Rutas a los catálogos de referencia, relativas a la raíz del proyecto
# (donde se ejecutan tanto los tests como el contenedor de la API).
_PRODUCTOS_PATH = os.path.join("dataset", "productos.csv")
_STOCK_PATH = os.path.join("dataset", "stock.csv")
_MOVIMIENTOS_PATH = os.path.join("dataset", "movimientos.csv")

_CATEGORICAL_FEATURES = [
    "gtin",
    "uso_principal",
    "linea_terapeutica",
    "canasta_vigente",
]

_NUMERIC_FEATURES = [
    "dia_semana",
    "dia_mes",
    "mes",
    "es_fin_de_mes",
    "stock",
    "gtin_media_historica",
    "gtin_dow_media_historica",
    "gtin_std_historica",
]

_MODEL_FEATURES = _CATEGORICAL_FEATURES + _NUMERIC_FEATURES


class ReplenishmentModel:

    def __init__(
        self
    ):
        self._model = None  # El modelo debe guardarse en este atributo.
        self._productos = None
        self._stock = None
        self._stock_last = None       # último stock conocido por gtin (fallback)
        self._global_mean = 0.0
        self._gtin_stats = None       # media/std histórica de consumo por gtin
        self._gtin_dow_stats = None   # media histórica de consumo por gtin + día de semana

    # ------------------------------------------------------------------
    # Catálogo de productos (dataset/productos.csv)
    # ------------------------------------------------------------------
    def _load_productos(self) -> pd.DataFrame:
        if self._productos is None:
            productos = pd.read_csv(_PRODUCTOS_PATH)
            productos["gtin"] = productos["gtin"].astype(str)
            productos["linea_terapeutica"] = productos["linea_terapeutica"].fillna(
                "SIN_CLASIFICAR"
            )
            self._productos = productos[
                ["gtin", "uso_principal", "linea_terapeutica", "canasta_vigente"]
            ]
        return self._productos

    # ------------------------------------------------------------------
    # Stock diario por producto (dataset/stock.csv). El nivel de stock en
    # el momento del movimiento es una señal de negocio real y disponible
    # (no depende del target ni causa data leakage: es información externa
    # observada independientemente de la salida que se quiere predecir).
    # ------------------------------------------------------------------
    def _load_stock(self) -> pd.DataFrame:
        if self._stock is None:
            stock = pd.read_csv(_STOCK_PATH)
            stock["gtin"] = stock["gtin"].astype(str)
            stock["fecha"] = pd.to_datetime(stock["fecha"], errors="coerce")
            stock = stock.dropna(subset=["fecha"])
            self._stock = stock[["gtin", "fecha", "stock"]]
            # Fallback para fechas fuera del rango cubierto por stock.csv
            # (p. ej. una fecha futura pedida a la API): último stock
            # conocido de cada producto.
            self._stock_last = (
                self._stock.sort_values("fecha")
                .groupby("gtin")["stock"]
                .last()
            )
        return self._stock

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------
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
        df = data.copy()

        # El negocio quiere predecir CONSUMO (salidas). Cuando el crudo trae
        # movimientos de entrada y salida (entrenamiento), nos quedamos solo
        # con las salidas; en modo servicio (API) este campo no viene, así
        # que no se aplica ningún filtro.
        if "tipo_movimiento" in df.columns:
            df = df[df["tipo_movimiento"] == "S"].drop(columns=["tipo_movimiento"])

        df["gtin"] = df["gtin"].astype(str)
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df = df.dropna(subset=["fecha", "gtin"]).reset_index(drop=True)

        # --- Features de calendario (estacionalidad) ---
        df["dia_semana"] = df["fecha"].dt.dayofweek
        df["dia_mes"] = df["fecha"].dt.day
        df["mes"] = df["fecha"].dt.month
        df["es_fin_de_mes"] = (
            df["fecha"].dt.days_in_month - df["dia_mes"] <= 3
        ).astype(int)

        # --- Features de contexto del producto ---
        df = df.merge(self._load_productos(), on="gtin", how="left")
        for col in ["uso_principal", "linea_terapeutica", "canasta_vigente"]:
            df[col] = df[col].fillna("DESCONOCIDO")

        # --- Feature de stock (nivel de inventario en la fecha del movimiento) ---
        self._load_stock()
        df = df.merge(self._stock, on=["gtin", "fecha"], how="left")
        # Si la fecha no está en stock.csv (p. ej. fecha futura pedida a la
        # API), se usa el último stock conocido del producto; si el gtin es
        # totalmente desconocido, se deja en 0.
        stock_fallback = df["gtin"].map(self._stock_last)
        df["stock"] = df["stock"].fillna(stock_fallback).fillna(0.0)

        # Nota: las features de tendencia histórica (media por gtin, media
        # por gtin+día de semana, etc.) NO se calculan aquí. Se agregan más
        # adelante, en fit()/predict(), a partir de estadísticas aprendidas
        # únicamente del set de entrenamiento (ver _enrich_with_history).
        # Calcularlas en preprocess() sería un error: esta función se llama
        # antes de que el modelo exista (p. ej. en modo servicio, donde el
        # payload de la API no trae "cantidad"), así que aquí solo dejamos
        # columnas "crudas" (identificadores + calendario + catálogo + stock).
        base_cols = (
            ["gtin", "fecha", "dia_semana", "dia_mes", "mes", "es_fin_de_mes", "stock"]
            + ["uso_principal", "linea_terapeutica", "canasta_vigente"]
        )
        features = df[base_cols].reset_index(drop=True)

        if target_column is not None:
            target = df[[target_column]].reset_index(drop=True)
            return features, target

        return features

    # ------------------------------------------------------------------
    # Enriquecimiento con estadísticas históricas (solo se calculan en fit,
    # con datos de entrenamiento; se aplican por igual en fit y en predict).
    # ------------------------------------------------------------------
    def _enrich_with_history(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        if self._gtin_stats is not None:
            df = df.merge(self._gtin_stats, on="gtin", how="left")
            df = df.merge(
                self._gtin_dow_stats, on=["gtin", "dia_semana"], how="left"
            )
        else:
            df["gtin_media_historica"] = np.nan
            df["gtin_std_historica"] = np.nan
            df["gtin_dow_media_historica"] = np.nan

        fallback_mean = self._global_mean if self._global_mean else 0.0
        df["gtin_media_historica"] = df["gtin_media_historica"].fillna(fallback_mean)
        df["gtin_std_historica"] = df["gtin_std_historica"].fillna(0.0)
        df["gtin_dow_media_historica"] = df["gtin_dow_media_historica"].fillna(
            df["gtin_media_historica"]
        )
        return df

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
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
        target_col = target.columns[0]
        train = features.copy()
        train[target_col] = target[target_col].values

        # Estadísticas históricas por producto (solo con datos de train, sin
        # fuga hacia el set de validación/test que use este mismo modelo).
        self._global_mean = float(train[target_col].mean())

        gtin_stats = (
            train.groupby("gtin")[target_col]
            .agg(gtin_media_historica="mean", gtin_std_historica="std")
            .reset_index()
        )
        gtin_stats["gtin_std_historica"] = gtin_stats["gtin_std_historica"].fillna(0.0)
        self._gtin_stats = gtin_stats

        gtin_dow_stats = (
            train.groupby(["gtin", "dia_semana"])[target_col]
            .mean()
            .reset_index()
            .rename(columns={target_col: "gtin_dow_media_historica"})
        )
        self._gtin_dow_stats = gtin_dow_stats

        # A partir de aquí self._gtin_stats/_gtin_dow_stats ya existen, así
        # que _enrich_with_history calcula las features de tendencia reales
        # (calculadas 100% sobre el fold de entrenamiento, sin fuga hacia
        # datos de validación/test que se le pasen luego a predict()).
        enriched = self._enrich_with_history(features)

        model_input = enriched[_MODEL_FEATURES]

        categorical_idx = [
            model_input.columns.get_loc(c) for c in _CATEGORICAL_FEATURES
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    _CATEGORICAL_FEATURES,
                ),
                ("num", "passthrough", _NUMERIC_FEATURES),
            ]
        )

        regressor = HistGradientBoostingRegressor(
            loss="absolute_error",  # alineado con la métrica de evaluación (MAE)
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            learning_rate=0.06,
            l2_regularization=1.0,
            categorical_features=categorical_idx,
            random_state=42,
        )

        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", regressor)]
        )
        pipeline.fit(model_input, target[target_col].values)

        self._model = pipeline

    # ------------------------------------------------------------------
    # Autoentrenamiento perezoso (usado por predict() cuando no hay un
    # modelo entrenado ni cargado todavía).
    # ------------------------------------------------------------------
    def _auto_train(self) -> None:
        movimientos = pd.read_csv(_MOVIMIENTOS_PATH)
        features, target = self.preprocess(
            data=movimientos, target_column="cantidad"
        )
        self.fit(features=features, target=target)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
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
        if self._model is None:
            # Autoentrenamiento perezoso: si nadie llamó fit()/load() antes,
            # se entrena una vez con el dataset canónico del proyecto. Esto
            # permite que predict() funcione "out of the box" (por ejemplo,
            # en la API, o en cualquier consumidor que solo quiera predecir
            # sin orquestar manualmente el pipeline de entrenamiento).
            self._auto_train()

        enriched = self._enrich_with_history(features)
        model_input = enriched[_MODEL_FEATURES]
        predictions = self._model.predict(model_input)
        # El consumo no puede ser negativo.
        predictions = np.clip(predictions, a_min=0, a_max=None)
        fechas = pd.to_datetime(features["fecha"]).dt.strftime("%Y-%m-%d")

        return [
            {"fecha": fecha, "cantidad": round(float(cantidad), 2)}
            for fecha, cantidad in zip(fechas, predictions)
        ]

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------
    def save(
        self,
        path: str
    ) -> None:
        """
        Guarda el modelo entrenado en disco.

        Args:
            path (str): ruta donde guardar el modelo.
        """
        state = {
            "model": self._model,
            "global_mean": self._global_mean,
            "gtin_stats": self._gtin_stats,
            "gtin_dow_stats": self._gtin_dow_stats,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(
        self,
        path: str
    ) -> None:
        """
        Carga un modelo entrenado desde disco.

        Args:
            path (str): ruta desde donde cargar el modelo.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self._model = state["model"]
        self._global_mean = state["global_mean"]
        self._gtin_stats = state["gtin_stats"]
        self._gtin_dow_stats = state["gtin_dow_stats"]