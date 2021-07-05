from typing import Tuple

from h5py import File
import numpy

from rna import RNA
from . import tools


def prediction(
    rna: RNA,
    rgb_path: str,
    mask_path: str
):
    bgr_image: numpy.ndarray = tools.load_BGR_image(rgb_path)
    YCbCr_image: numpy.ndarray = tools.BGR_to_YCbCr(bgr_image)
    Cr: numpy.ndarray = YCbCr_image[:,:,1].flatten()
    Cb: numpy.ndarray = YCbCr_image[:,:,2].flatten()
    X0: numpy.ndarray = numpy.vstack((Cb, Cr))
    
    predicted_mask: numpy.ndarray = rna.predict(X0)

    tools.save_image(
        filename=mask_path,
        image=predicted_mask.reshape(
            YCbCr_image[:,:,0].shape
        )
    )
