import numpy
import json
from matplotlib import pyplot
from typing import (
    Dict,
    Any,
    Callable
)


def plot_prediccion(
    data_path: str, expr: str, fn: Callable[[numpy.ndarray], numpy.ndarray], filename: str, ext: str = 'svg'
):
    with open(data_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    
    x0: numpy.ndarray = numpy.array(data["x0"])
    Y: numpy.ndarray = numpy.array(data["Y"])

    fig = pyplot.figure()
    pyplot.plot(x0[0,:], Y[0,:], 'g+', label='predicción')
    x0 = numpy.array(sorted(x0[0,:]))
    pyplot.plot(x0, fn(x0), '-', color='b', label=expr)
    pyplot.xlabel('x')
    pyplot.ylabel('Y')
    pyplot.legend()
    pyplot.savefig(f"{filename}.{ext}")