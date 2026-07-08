"""
Script para entrenar el modelo y persistirlo en disco.

Usage:
    python -m challenge.train
"""

from challenge.model import ReplenishmentModel
import pandas as pd


def main():
    model = ReplenishmentModel()

    # Cargar datos
    movimientos = pd.read_csv("dataset/movimientos.csv")

    # TODO: Preprocesar y entrenar
    # features, target = model.preprocess(data=movimientos, target_column="cantidad")
    # model.fit(features=features, target=target)

    # TODO: Persistir modelo
    # model.save("model.pkl")


if __name__ == "__main__":
    main()