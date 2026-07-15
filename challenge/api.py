import os
from datetime import datetime
from typing import List

import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

from challenge.model import ReplenishmentModel

app = fastapi.FastAPI()

# Rutas relativas a la raíz del proyecto (mismo criterio que challenge/model.py:
# funcionan tanto al correr pytest desde la raíz como en el contenedor Docker,
# cuyo WORKDIR es /app).
_MODEL_PATH = "model.pkl"
_MOVIMIENTOS_PATH = os.path.join("dataset", "movimientos.csv")
_PRODUCTOS_PATH = os.path.join("dataset", "productos.csv")


def _get_or_train_model() -> ReplenishmentModel:
    """
    Carga el modelo ya entrenado desde disco (generado con `python -m
    challenge.train`). Si no existe (p. ej. primer arranque sin ese paso
    previo), lo entrena on-the-fly con el dataset y lo persiste, para que
    la API quede funcional igual.
    """
    model = ReplenishmentModel()

    if os.path.exists(_MODEL_PATH):
        model.load(_MODEL_PATH)
        return model

    movimientos = pd.read_csv(_MOVIMIENTOS_PATH)
    features, target = model.preprocess(data=movimientos, target_column="cantidad")
    model.fit(features=features, target=target)
    model.save(_MODEL_PATH)
    return model


def _load_valid_gtins() -> set:
    productos = pd.read_csv(_PRODUCTOS_PATH)
    return set(productos["gtin"].astype(str))


# Se inicializan al importar el módulo (no en un evento de "startup" de
# FastAPI) para que queden listos sin depender de cómo el test runner
# gestione el ciclo de vida de la app.
_model = _get_or_train_model()
_valid_gtins = _load_valid_gtins()


class ProductRequest(BaseModel):
    gtin: str
    fecha: str


class PredictRequest(BaseModel):
    products: List[ProductRequest]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    if not request.products:
        raise HTTPException(
            status_code=400,
            detail="Debe enviar al menos un producto en 'products'."
        )

    rows = []
    for product in request.products:
        gtin = str(product.gtin)

        # Validación de fecha
        try:
            fecha = datetime.strptime(product.fecha, "%Y-%m-%d")
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Fecha invalida: '{product.fecha}'. "
                    "Formato esperado YYYY-MM-DD."
                )
            )

        # Validación de producto conocido
        if gtin not in _valid_gtins:
            raise HTTPException(
                status_code=400,
                detail=f"Producto desconocido: '{gtin}'."
            )

        rows.append({"gtin": gtin, "fecha": fecha.strftime("%Y-%m-%d")})

    data = pd.DataFrame(rows)
    features = _model.preprocess(data=data)
    predictions = _model.predict(features=features)

    predict_result = [
        {
            "gtin": row["gtin"],
            "fecha": pred["fecha"],
            "cantidad": pred["cantidad"],
        }
        for row, pred in zip(rows, predictions)
    ]

    return {"predict": predict_result}