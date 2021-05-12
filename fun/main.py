from typer import Typer
import numpy
import json
from h5py import File
from typing import Dict, Any
from src.rna import RNA
from scripts.entrenamiento import train_rna
from scripts.prediccion import prediction
from scripts.preparar_datos import prepare_data


app = Typer()

@app.command(name="prepare-training-data")
def prepare_training_data(specs_path: str):
    with open(specs_path) as f:
        specs: Dict[str, Any] = json.load(f)
    prepare_data(specs)

@app.command(name="train")
def train(
    training_data_path: str,
    trained_rna_path: str,
    hidden_neurons: int = 5,
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

@app.command(name="prediction")
def predict(trained_rna_path: str, input_data_path: str):
    with open(input_data_path, "r") as f:
        x0: numpy.ndarray = numpy.array(json.load(f)["x0"])
    rna_parameters = {}
    with File(trained_rna_path, "r") as data:
        rna_parameters["W2"] = data["parametros_finales"]["W2"][()]
        rna_parameters["W1"] = data["parametros_finales"]["W1"][()]
        rna_parameters["b1"] = data["parametros_finales"]["b1"][()]
        rna_parameters["neuronas_ocultas"] = int(data["arq"]["neuronas_ocultas"][()])
    rna = RNA.from_previous_rna(parametros_de_red=rna_parameters)
    pred: numpy.ndarray = prediction(rna=rna, x0=x0)
    out_data = {
        "x0": x0.tolist(),
        "preds": pred.tolist()
    }
    with open("preds_20_000.json", "w") as f:
        f.write(json.dumps(out_data))

if __name__ == "__main__":
    app()