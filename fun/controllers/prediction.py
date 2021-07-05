import numpy

from rna import RNA


def prediction(rna: RNA, x0: numpy.ndarray) -> numpy.ndarray:
    return rna.predict(x0)