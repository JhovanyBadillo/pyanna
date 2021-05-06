from typing import Tuple, List
import numpy
import glob
from . import tools
from tqdm import tqdm

def prepare_data(ruta_de_imagenes_rgb: str, ruta_de_mascaras: str):
    ruta_de_imagenes_rgb: str = ruta_de_imagenes_rgb
    ruta_de_mascaras: str = ruta_de_mascaras
    imagenes_rgb: List[str] = glob.glob(f"{ruta_de_imagenes_rgb}/*")

    STD_SIZE: Tuple[int] = (88, 72)

    imagenes_procesadas: List[numpy.ndarray] = []
    mascaras_procesadas: List[numpy.ndarray] = []

    imagenes_rgb: tqdm = tqdm(
        iterable=imagenes_rgb,
        desc="Preparando conjunto de entrenamiento",
        total=len(imagenes_rgb),
        unit="imagen"
    )
    for imagen in imagenes_rgb:
        nombre_de_archivo: str = imagen.split('/')[-1]
        imagen_bgr: numpy.ndarray = tools.load_BGR_image(
            filename=f"{ruta_de_imagenes_rgb}/{nombre_de_archivo}"
        )
        mascara: numpy.ndarray = tools.load_mask(
            filename=f"{ruta_de_mascaras}/{nombre_de_archivo}"
        )
        imagen_YCbCr: numpy.ndarray = tools.BGR_to_YCbCr(
            bgr_image=imagen_bgr
        )
        imagen_YCbCr_redimensionada: numpy.ndarray = tools.resize(
            image=imagen_YCbCr,
            new_size=STD_SIZE
        )
        mascara_redimensionada: numpy.ndarray = tools.resize(
            image=mascara,
            new_size=STD_SIZE
        )
        mascara_normalizada: numpy.ndarray = (mascara_redimensionada/255).astype(dtype="uint8")
        mascara_transformada: numpy.ndarray = mascara_normalizada.flatten()
        Cr: numpy.ndarray = imagen_YCbCr_redimensionada[:,:,1].flatten()
        Cb: numpy.ndarray = imagen_YCbCr_redimensionada[:,:,2].flatten()
        imagen_transformada: numpy.ndarray = numpy.vstack((Cb, Cr))

        imagenes_procesadas.append(imagen_transformada)
        mascaras_procesadas.append(mascara_transformada)

    X: numpy.ndarray = numpy.vstack(imagenes_procesadas)
    Z: numpy.ndarray = numpy.vstack(mascaras_procesadas)

    tools.save_data(
        filename="datos_de_entrenamiento.hdf5", 
        data={"X": X, "Z": Z}
    )
