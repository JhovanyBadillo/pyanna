from typing import (
    Tuple,
    Dict,
    Union,
    Callable
)

import numpy
import sympy

from . import tools


def prepare_data(
    specs: Dict[str, Union[str, int]],
    training_data_path: str
):
    symbols: Tuple[sympy.Symbol] = (sympy.Symbol(specs['var']),)
    expr: str = specs['function']
    fn: Callable[[numpy.ndarray], numpy.ndarray] = sympy.lambdify(
        args=symbols,
        expr=expr,
        modules='numpy'
    )

    X: numpy.ndarray = numpy.random.rand(1, int(specs['samples']))
    Z: numpy.ndarray = fn(X)

    tools.save_data(
        filename=training_data_path, 
        data={"X": X, "Z": Z}
    )
