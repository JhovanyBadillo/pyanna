from typing import Optional

from typer import Typer
from h5py import File

from rna import RNA
from controllers import *


app = Typer()

@app.command(name="prepare-training-data")
def prepare_training_data(
    rgb_path: str = "./rgb",
    masks_path: str = "./masks",
    training_data_path: str = "training_data.hdf5"
):
    prepare_data(
        ruta_de_imagenes_rgb=rgb_path,
        ruta_de_mascaras=masks_path,
        training_data_path=training_data_path
    )


@app.command(name="train")
def train(
    training_data_path: str = "training_data.hdf5",
    trained_rna_path: str = "trained_rna.hdf5",
    hidden_neurons: int = 25,
    epochs: int = 10_000,
    learning_rate: float = 0.2
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
    rgb_path: str,
    mask_path: str,
    trained_rna_path: str = "trained_rna.hdf5"
):
    rna_parameters = {}

    with File(trained_rna_path, "r") as data:
        rna_parameters["W2"] = data["parametros_finales"]["W2"][()]
        rna_parameters["W1"] = data["parametros_finales"]["W1"][()]
        rna_parameters["b1"] = data["parametros_finales"]["b1"][()]
        rna_parameters["neuronas_ocultas"] = int(data["arq"]["neuronas_ocultas"][()])

    rna = RNA.from_previous_rna(parametros_de_red=rna_parameters)
    prediction(rna=rna, rgb_path=rgb_path, mask_path=mask_path)


if __name__ == "__main__":
    app()