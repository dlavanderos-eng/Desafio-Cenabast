import os
from datetime import datetime
from typing import List

import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import ReplenishmentModel

app = fastapi.FastAPI()

_MODEL_PATH = "model.pkl"
_MOVIMIENTOS_PATH = os.path.join("dataset", "movimientos.csv")
_PRODUCTOS_PATH = os.path.join("dataset", "productos.csv")
_DATE_FORMAT = "%Y-%m-%d"


def _load_or_train_model() -> ReplenishmentModel:
    """
    Carga el modelo ya entrenado desde disco (model.pkl, generado por
    `python -m challenge.train`). Si no existe (p. ej. primer arranque en
    un contenedor limpio), lo entrena al vuelo con el dataset incluido en
    la imagen, para que la API siempre quede lista para servir.
    """
    model = ReplenishmentModel()
    try:
        model.load(_MODEL_PATH)
    except Exception:
        movimientos = pd.read_csv(_MOVIMIENTOS_PATH)
        features, target = model.preprocess(data=movimientos, target_column="cantidad")
        model.fit(features=features, target=target)
        try:
            model.save(_MODEL_PATH)
        except OSError:
            # Si el filesystem es de solo lectura (algunos entornos de
            # Cloud Run), simplemente seguimos con el modelo en memoria.
            pass
    return model


def _load_known_gtins() -> set:
    productos = pd.read_csv(_PRODUCTOS_PATH)
    return set(productos["gtin"].astype(str))


_model = _load_or_train_model()
_known_gtins = _load_known_gtins()


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
    if len(request.products) == 0:
        raise fastapi.HTTPException(
            status_code=400, detail="Debe enviar al menos un producto."
        )

    for product in request.products:
        if product.gtin not in _known_gtins:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Producto desconocido: {product.gtin}",
            )
        try:
            datetime.strptime(product.fecha, _DATE_FORMAT)
        except ValueError:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Fecha inválida: {product.fecha}. Formato esperado: YYYY-MM-DD.",
            )

    data = pd.DataFrame(
        [{"gtin": p.gtin, "fecha": p.fecha} for p in request.products]
    )

    features = _model.preprocess(data=data)
    predictions = _model.predict(features=features)

    predict = [
        {
            "gtin": gtin,
            "fecha": pred["fecha"],
            "cantidad": pred["cantidad"],
        }
        for gtin, pred in zip(features["gtin"].tolist(), predictions)
    ]

    return {"predict": predict}