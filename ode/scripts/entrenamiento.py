import numpy
import json
from h5py import File
from typing import Dict, Any
from .tools import save_data, plot_costs


def train_rna(
    rna,
    ode,
    points: int,
    trained_rna_path: str,
    epochs: int,
    learning_rate: float
):
    R: int = points
    x0: numpy.ndarray = numpy.random.rand(1, R)

    parametros_iniciales: Dict[str, numpy.ndarray] = \
        rna.obtener_parametros_de_red()

    rna.fit(x0=x0, ode=ode)

    rna.train(epocas=epochs, eta=learning_rate)

    plot_costs(rna.costs_collection)

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

    save_data(trained_rna_path, metadatos)

    with open("aproximacion.json", "w") as f:
        f.write(json.dumps({
            "x0": rna.x0.tolist(),
            "Y": rna.Y.tolist()
        }))