from h5py import File
import numpy
from typing import Tuple
from . import tools


def inference(rna, rgb_path: str):
    bgr_image: numpy.ndarray = tools.load_BGR_image(rgb_path)
    YCbCr_image: numpy.ndarray = tools.BGR_to_YCbCr(bgr_image)
    # resized_YCbCr_image: numpy.ndarray = resize(
    #     image=YCbCr_image,
    #     new_size=(1040, 1386)
    # )
    Cr: numpy.ndarray = YCbCr_image[:,:,1].flatten()
    Cb: numpy.ndarray = YCbCr_image[:,:,2].flatten()
    X0: numpy.ndarray = numpy.vstack((Cb, Cr))
    
    predicted_mask: numpy.ndarray = rna.predict(X0)

    tools.save_image(
        filename=f"{rgb_path.split('/')[-1].split('.')[0]}_mask.png",
        image=predicted_mask.reshape(
            YCbCr_image[:,:,0].shape
        )
    )