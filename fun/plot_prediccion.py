import numpy
import json
from matplotlib import pyplot
from typing import Dict, Any


def plot_prediccion(data_path: str, expr: str, filename: str, ext: str = 'svg'):
    with open(data_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    
    x0: numpy.ndarray = numpy.array(data["x0"])
    Z: numpy.ndarray = numpy.array(data["preds"])

    fig = pyplot.figure()
    pyplot.plot(x0[0,:], Z[0,:], 'g+', label='predicción')
    pyplot.plot(sorted(x0[0,:]), numpy.exp(-numpy.array(sorted(x0[0,:]))), '-', color='b', label=expr)
    pyplot.xlabel('x')
    pyplot.ylabel('Z')
    pyplot.legend()
    pyplot.savefig(f"{filename}.{ext}")