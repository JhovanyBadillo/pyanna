import numpy


def prediction(rna, x0: numpy.ndarray) -> numpy.ndarray:
    return rna.predict(x0)