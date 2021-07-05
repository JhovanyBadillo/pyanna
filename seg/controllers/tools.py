import cv2
import numpy
import h5py
from typing import (
    Tuple,
    Dict,
    Any,
    List
)
from matplotlib import pyplot


def load_BGR_image(filename: str) -> numpy.ndarray:
    return cv2.imread(filename=filename)

def load_mask(filename: str) -> numpy.ndarray:
    return cv2.imread(filename=filename, flags=cv2.IMREAD_GRAYSCALE)

def save_image(filename: str, image: numpy.ndarray):
    cv2.imwrite(filename=filename, img=image)

def segment_with_mask(bgr_image: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    return cv2.bitwise_and(bgr_image, bgr_image, mask=mask)

def BGR_to_YCbCr(bgr_image: numpy.ndarray) -> numpy.ndarray:
    return cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2YCrCb)

def resize(image: numpy.ndarray, new_size: Tuple[int]) -> numpy.ndarray:
    return cv2.resize(image, new_size)

def save_data(filename: str, data: Dict[str, Any]):
    with h5py.File(name=filename, mode="w") as f:
        for k, v in data.items():
            if isinstance(v, dict):
                subgroup = f.create_group(k)
                for subk, subv in data[k].items():
                    subgroup.create_dataset(name=subk, data=subv)
            else:
                f.create_dataset(name=k, data=v)

def plot_costs(costs: List[float]):
    fig, ax = pyplot.subplots()
    ax.plot(range(len(costs)), costs, 'b+')
    ax.set(xlabel='epoch', ylabel='cost', title='cost per epoch')
    pyplot.savefig("costs.png", dpi=400)