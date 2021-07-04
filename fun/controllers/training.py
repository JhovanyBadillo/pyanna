from typing import Dict, Any

import numpy
from h5py import File

from . import tools


def train_rna(
    rna,
    training_data_path: str,
    trained_rna_path: str,
    epochs: int,
    learning_rate: float
):
    datos_de_entrenamiento: File = \
        File(training_data_path, "r")

    X: numpy.ndarray = datos_de_entrenamiento["X"][()]
    Z: numpy.ndarray = datos_de_entrenamiento["Z"][()]

    parametros_iniciales: Dict[str, numpy.ndarray] = \
        rna.obtener_parametros_de_red()

    rna.fit(X=X, Z=Z)

    rna.train(epocas=epochs, eta=learning_rate)

    tools.plot_costs(rna.costs_collection)

    parametros_finales: Dict[str, numpy.ndarray] = \
        rna.obtener_parametros_de_red()

    metadatos: Dict[str, Any] = {
        "parametros_iniciales": {**parametros_iniciales},
        "parametros_finales": {**parametros_finales},
        "arq": {
            "neuronas_ocultas": rna.neuronas_ocultas,
        },
        "entrenamiento": {
            "eta": rna.eta,
            "epocas": epochs
        },
        "costos": rna.costs_collection
    }

    tools.save_data(trained_rna_path, metadatos)