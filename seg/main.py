import typer
from h5py import File
from src.rna import RNA
from scripts.preparar_datos import prepare_data
from scripts.entrenamiento import train_rna
from scripts.inferencia import inference

app = typer.Typer()

@app.command(name="prepare-training-data")
def prepare_training_data(rgb_path: str, masks_path: str):
    prepare_data(
        ruta_de_imagenes_rgb=rgb_path,
        ruta_de_mascaras=masks_path
    )

@app.command(name="train")
def train(
    training_data_path: str,
    trained_rna_path: str,
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

@app.command(name="inference")
def predict(trained_rna_path: str, rgb_path: str):
    rna_parameters = {}
    with File(trained_rna_path, "r") as data:
        rna_parameters["W2"] = data["parametros_finales"]["W2"][()]
        rna_parameters["W1"] = data["parametros_finales"]["W1"][()]
        rna_parameters["b1"] = data["parametros_finales"]["b1"][()]
        rna_parameters["neuronas_ocultas"] = int(data["arq"]["neuronas_ocultas"][()])
    rna = RNA.from_trained_rna(parametros_de_red=rna_parameters)
    inference(rna=rna, rgb_path=rgb_path)

if __name__ == "__main__":
    app()