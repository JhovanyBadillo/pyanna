from typing import (
    Dict,
    Union
)

import numpy
from tqdm import tqdm


class RNA:
    def __init__(self, neuronas_ocultas: int):
        self._neuronas_ocultas: int = neuronas_ocultas
        self.W1: numpy.ndarray = numpy.random.rand(1, self.neuronas_ocultas)
        self.W2: numpy.ndarray = numpy.random.rand(self.neuronas_ocultas, 1)
        self.b1: numpy.ndarray = numpy.random.rand(self.neuronas_ocultas, 1)
        self.costs_collection: List[float] = []


    @classmethod
    def from_previous_rna(
        cls,
        parametros_de_red: Dict[str, Union[numpy.ndarray, int]]
    ):
        rna_previa = cls(
            neuronas_ocultas=\
                parametros_de_red["neuronas_ocultas"]
        )
        rna_previa.W1 = parametros_de_red["W1"]
        rna_previa.W2 = parametros_de_red["W2"]
        rna_previa.b1 = parametros_de_red["b1"]

        return rna_previa


    @property
    def neuronas_ocultas(self) -> int:
        return self._neuronas_ocultas

    @property
    def R(self) -> int:
        return self._R
    
    @R.setter
    def R(self, R: int):
        self._R: int = R

    @property
    def x0(self) -> numpy.ndarray:
        return self._x0
    
    @x0.setter
    def x0(self, x0: numpy.ndarray):
        self._x0: numpy.ndarray = x0

    @property
    def Z(self) -> numpy.ndarray:
        return self._Z
    
    @Z.setter
    def Z(self, Z: numpy.ndarray):
        self._Z: numpy.ndarray = Z
    
    @property
    def eta(self) -> float:
        return self._eta
    
    @eta.setter
    def eta(self, eta: float):
        self._eta: float = eta

    @staticmethod
    def sigmoide(X: numpy.ndarray) -> numpy.ndarray:
        return 1.0/(1.0 + numpy.exp(-X))

    @staticmethod
    def derivada_sigmoide(X: numpy.ndarray) -> numpy.ndarray:
        return RNA.sigmoide(X)*(1.0 - RNA.sigmoide(X))


    def obtener_parametros_de_red(self) -> Dict[str, numpy.ndarray]:
        parametros_de_red: Dict = {
            "W1": self.W1,
            "W2": self.W2,
            "b1": self.b1
        }
        return parametros_de_red

    def fit(self, X: numpy.ndarray, Z: numpy.ndarray):
        self.R = Z.shape[1]
        self.x0 = X
        self.Z = Z

        self.B1: numpy.ndarray = numpy.dot(
            self.b1,
            numpy.ones((1, self.R))
        )

    def propagar(self):
        self.S1: numpy.ndarray = numpy.dot(
            self.W1.T,
            self.x0
        ) - self.B1

        self.X1: numpy.ndarray = RNA.sigmoide(self.S1)

        self.Y: numpy.ndarray = numpy.dot(
            self.W2.T,
            self.X1
        )
    
    def retropropagar(self):
        self.vector_costos: numpy.ndarray = self.Y - self.Z

        self.costo_total: numpy.ndarray = (1.0/(2.0*self.R))*numpy.dot(
            self.vector_costos,
            self.vector_costos.T
        )

        self.costs_collection.append(float(self.costo_total))

        self.delta2: numpy.ndarray = (1.0/self.R)*self.vector_costos

        self.dC_dW2: numpy.ndarray = numpy.dot(
            self.delta2,
            self.X1.T
        )
        self.delta1: numpy.ndarray = numpy.multiply(
            numpy.dot(self.W2, self.delta2),
            RNA.derivada_sigmoide(self.S1)
        )

        self.dC_dW1: numpy.ndarray = numpy.dot(
            self.x0,
            self.delta1.T
        )

        self.dC_db1: numpy.ndarray = -numpy.dot(
            numpy.ones((1, self.R)),
            self.delta1.T
        )

    def corregir(self):
        self.b1 = self.b1 - self.eta*self.dC_db1.T
        self.W1 = self.W1 - self.eta*self.dC_dW1
        self.W2 = self.W2 - self.eta*self.dC_dW2.T

    def train(self, epocas: int, eta: float=0.2):
        self.eta = eta
        epocas = tqdm(
            iterable=range(epocas),
            desc="Training RNA",
            total=epocas,
            unit="epoch"
        ) 
        for epoca in epocas:
            self.propagar()
            self.retropropagar()
            self.corregir()

    def predict(self, x0: numpy.ndarray) -> numpy.ndarray:
        assert len(x0.shape) == 2
        self.x0 = x0
        self.R = self.x0.shape[1]
        self.B1 = numpy.dot(
            self.b1,
            numpy.ones((1, self.R))
        )
        self.propagar()
        return self.Y
