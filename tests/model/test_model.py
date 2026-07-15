import unittest
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from challenge.model import ReplenishmentModel


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = ReplenishmentModel()
        self.data = pd.read_csv(filepath_or_buffer="dataset/movimientos.csv")

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="cantidad"
        )

        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.DataFrame)

    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)

    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="cantidad"
        )

        self.model.fit(
            features=features,
            target=target
        )

        assert self.model._model is not None

    def test_model_predict(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        predicted = self.model.predict(
            features=features
        )

        assert isinstance(predicted, list)
        assert len(predicted) > 0
        for pred in predicted:
            assert "fecha" in pred
            assert "cantidad" in pred

    def test_model_predict_all_products(
        self
    ):
        # El modelo debe predecir para todos los productos del dataset
        features = self.model.preprocess(
            data=self.data
        )

        predicted = self.model.predict(
            features=features
        )

        assert len(predicted) == features.shape[0]

    def test_model_performance(
        self
    ):
        # El modelo debe rendir mejor que un baseline por producto
        features, target = self.model.preprocess(
            data=self.data,
            target_column="cantidad"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        self.model.fit(
            features=X_train,
            target=y_train
        )

        predictions = self.model.predict(
            features=X_test
        )

        # Las predicciones deben cubrir todas las muestras del set de test
        assert len(predictions) == len(X_test)

        predicted_cantidades = [p["cantidad"] for p in predictions]

        # Baseline: predecir la media por producto del target de entrenamiento
        train_means = pd.concat(
            [X_train[["gtin"]].reset_index(drop=True),
             y_train.reset_index(drop=True)], axis=1
        )
        product_means = train_means.groupby("gtin")["cantidad"].mean().to_dict()
        global_mean = y_train.mean().iloc[0]
        baseline_preds = [
            product_means.get(g, global_mean)
            for g in X_test["gtin"].values
        ]
        baseline_mae = mean_absolute_error(y_test, baseline_preds)

        model_mae = mean_absolute_error(y_test, predicted_cantidades)

        # El modelo debe ser al menos 25% mejor que el baseline por producto
        assert model_mae < baseline_mae * 0.75

    def test_model_save_and_load(
        self
    ):
        import os
        import tempfile

        features, target = self.model.preprocess(
            data=self.data,
            target_column="cantidad"
        )
        self.model.fit(features=features, target=target)

        # Guardar
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        self.model.save(tmp_path)

        # Cargar en un nuevo modelo
        new_model = ReplenishmentModel()
        new_model.load(tmp_path)

        assert new_model._model is not None

        # Limpiar
        os.unlink(tmp_path)