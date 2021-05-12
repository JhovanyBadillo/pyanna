from typing import (
    Tuple,
    Dict,
    Any,
    Callable
)
import numpy
from . import tools
import sympy

def prepare_data(specs: Dict[str, Any]):
    symbols: Tuple[sympy.Symbol] = (sympy.Symbol(specs['var']),)
    expr: str = specs['function']
    fn: Callable[numpy.ndarray, numpy.ndarray] = sympy.lambdify(
        args=symbols,
        expr=expr,
        modules='numpy'
    )

    X: numpy.ndarray = numpy.random.rand(1, int(specs['samples']))
    Z: numpy.ndarray = fn(X)

    tools.save_data(
        filename="datos_de_entrenamiento.hdf5", 
        data={"X": X, "Z": Z}
    )
