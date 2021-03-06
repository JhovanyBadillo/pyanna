from typing import (
    Tuple,
    List,
    Dict,
    Any
)

import numpy
from h5py import File
from tqdm import tqdm


class RNA:
    def __init__(self, neuronas_ocultas: int):
        self._neuronas_ocultas: int = neuronas_ocultas
        self.W1: numpy.ndarray = numpy.random.rand(2, self.neuronas_ocultas)
        self.W2: numpy.ndarray = numpy.random.rand(self.neuronas_ocultas, 1)
        self.b1: numpy.ndarray = numpy.random.rand(self.neuronas_ocultas, 1)
        self._L: int = None
        self._R: int = None
        self._J: int = None
        self._X0: numpy.ndarray = None
        self._Z: numpy.ndarray = None
        self._B1: numpy.ndarray = None
        self.costs_collection: List[float] = []

    
    @classmethod
    def from_previous_rna(cls, parametros_de_red: Dict[str, numpy.ndarray]):
        rna_entrenada = cls(
            neuronas_ocultas=\
                parametros_de_red["neuronas_ocultas"]
        )
        rna_entrenada.W1 = parametros_de_red["W1"]
        rna_entrenada.W2 = parametros_de_red["W2"]
        rna_entrenada.b1 = parametros_de_red["b1"]

        return rna_entrenada


    @property
    def neuronas_ocultas(self) -> int:
        return self._neuronas_ocultas

    @property
    def L(self) -> int:
        if self._L is None:
            raise Exception("Attribute _L is None")
        return self._L
    
    @L.setter
    def L(self, L: int):
        self._L = L
    
    @property
    def R(self) -> int:
        if self._R is None:
            raise Exception("Attribute _R is None")
        return self._R
    
    @property
    def J(self) -> int:
        if self._J is None:
            raise Exception("Attribute _J is None")
        return self._J
    
    @property
    def X0(self) -> numpy.ndarray:
        if self._X0 is None:
            raise Exception("Attribute _X0 is None")
        return self._X0
    
    @X0.setter
    def X0(self, X0: numpy.ndarray):
        self._X0 = X0
    
    @property
    def Z(self) -> numpy.ndarray:
        if self._Z is None:
            raise Exception("Attribute _Z is None")
        return self._Z

    @property
    def B1(self) -> numpy.ndarray:
        if self._B1 is None:
            raise Exception("Attribute _B1 is None")
        return self._B1
    
    @B1.setter
    def B1(self, B1: numpy.ndarray):
        self._B1 = B1
    
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
        self._R = Z.shape[0]
        self._L = Z.shape[1]
        self._J = self.L*self.R
        self._X0 = numpy.hstack(
            numpy.split(X, self.R)
        )/255.0
        self._Z = numpy.hstack(
            numpy.split(Z, self.R)
        )

        self._B1: numpy.ndarray = numpy.dot(
            self.b1,
            numpy.ones((1, self.J))
        )

    def propagar(self):
        self.S1: numpy.ndarray = numpy.dot(
            self.W1.T,
            self.X0
        ) - self.B1

        self.X1: numpy.ndarray = RNA.sigmoide(self.S1)

        self.Y: numpy.ndarray = numpy.dot(
            self.W2.T,
            self.X1
        )
    
    def retropropagar(self):
        self.vector_costos: numpy.ndarray = self.Y - self.Z

        self.costo_total: numpy.ndarray = (1.0/(2.0*self.J))*numpy.dot(
            self.vector_costos,
            self.vector_costos.T
        )

        self.costs_collection.append(float(self.costo_total))

        self.delta2: numpy.ndarray = (1.0/(self.J))*self.vector_costos

        self.dC_dW2: numpy.ndarray = numpy.dot(
            self.delta2,
            self.X1.T
        )

        self.delta1: numpy.ndarray = numpy.multiply(
            numpy.dot(self.W2, self.delta2),
            RNA.derivada_sigmoide(self.S1)
        )

        self.dC_dW1: numpy.ndarray = numpy.dot(
            self.X0,
            self.delta1.T
        )

        self.dC_db1: numpy.ndarray = -numpy.dot(
            numpy.ones((1, self.J)),
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

    def predict(self, X0: numpy.ndarray) -> numpy.ndarray:
        assert len(X0.shape) == 2
        beta: float = 0.3
        self.X0 = X0/255.0
        self.L = self.X0.shape[1]
        self.B1 = numpy.dot(
            self.b1,
            numpy.ones((1, self.L))
        )
        self.propagar()
        return numpy.where(self.Y > beta, 255, 0)
