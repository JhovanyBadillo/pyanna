import json
from typing import Dict, Union

from typer import Typer
import numpy
import h5py 

from rna import RNA
from controllers import *


app = Typer()

@app.command(name="prepare-training-data")
def prepare_training_data(
    specs_path: str = "fun_specs.json",
    training_data_path: str = "training_data.hdf5"
):
    with open(specs_path) as f:
        specs: Dict[str, Union[str, int]] = json.load(f)
    prepare_data(specs, training_data_path)


@app.command(name="train")
def train(
    training_data_path: str = "training_data.hdf5",
    trained_rna_path: str = "trained_rna.hdf5",
    hidden_neurons: int = 3,
    epochs: int = 20_000,
    learning_rate: float = 0.5
):
    rna = RNA(neuronas_ocultas=hidden_neurons)
    train_rna(
        rna=rna,
        training_data_path=training_data_path,
        trained_rna_path=trained_rna_path,
        epochs=epochs,
        learning_rate=learning_rate
    )


@app.command(name="predict")
def predict(
    input_data_path: str,
    output_data_path: str,
    trained_rna_path: str = "trained_rna.hdf5"
):
    with open(input_data_path, "r") as f:
        x0: numpy.ndarray = numpy.array(json.load(f)["x0"])

    rna_parameters = {}

    with h5py.File(trained_rna_path, "r") as data:
        rna_parameters["W2"] = data["parametros_finales"]["W2"][()]
        rna_parameters["W1"] = data["parametros_finales"]["W1"][()]
        rna_parameters["b1"] = data["parametros_finales"]["b1"][()]
        rna_parameters["neuronas_ocultas"] = int(
            data["arq"]["neuronas_ocultas"][()]
        )

    rna = RNA.from_previous_rna(parametros_de_red=rna_parameters)
    pred: numpy.ndarray = prediction(rna=rna, x0=x0)
    out_data = {
        "x0": x0.tolist(),
        "Y": pred.tolist()
    }

    with open(output_data_path, "w") as f:
        f.write(json.dumps(out_data))


if __name__ == "__main__":
    app()