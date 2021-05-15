from typer import Typer
import numpy
import json
from h5py import File
from typing import Dict, Any
from src.ode_rna import RnaForOde, Ode
from scripts.entrenamiento import train_rna
from scripts.prediccion import prediction


app = Typer()

@app.command(name="train")
def train(
    points: int = 100,
    trained_rna_path: str = "rna_entrenada.hdf5",
    hidden_neurons: int = 3,
    epochs: int = 10_000,
    learning_rate: float = 0.2
):
    rna = RnaForOde(neuronas_ocultas=hidden_neurons)
    ode = Ode(
        x0_0=0.0,
        A=0.0,
        F=lambda X, Y: -Y + numpy.exp(-X),
        dF=lambda X, Y: -numpy.ones((1, points))
    )
    train_rna(
        rna=rna,
        ode=ode,
        points=points,
        trained_rna_path=trained_rna_path,
        epochs=epochs,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    app()