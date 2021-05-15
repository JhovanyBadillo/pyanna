import numpy
from math import sqrt
from pydantic import BaseModel
from typing import (
    Callable,
    Dict,
    Union
)
from tqdm import tqdm


class Ode(BaseModel):
    x0_0: float
    A: float
    F: Callable
    dF: Callable


class RnaForOde:
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
        return RnaForOde.sigmoide(X)*(1.0 - RnaForOde.sigmoide(X))

    @staticmethod
    def segunda_derivada_sigmoide(X: numpy.ndarray) -> numpy.ndarray:
        return RnaForOde.derivada_sigmoide(X)*(1.0 - 2.0*RnaForOde.sigmoide(X))


    def obtener_parametros_de_red(self) -> Dict[str, numpy.ndarray]:
        parametros_de_red: Dict = {
            "W1": self.W1,
            "W2": self.W2,
            "b1": self.b1
        }
        return parametros_de_red

    def fit(self, x0: numpy.ndarray, ode: Ode):
        assert len(x0.shape) == 2
        self.R = x0.shape[1]
        self.x0 = numpy.hstack((
            [[ode.x0_0,]], x0
        ))
        self.ode = ode


        self.B1: numpy.ndarray = numpy.dot(
            self.b1,
            numpy.ones((1, self.R + 1))
        )

    def propagar(self):
        self.S1: numpy.ndarray = numpy.dot(
            self.W1.T,
            self.x0
        ) - self.B1

        self.X1: numpy.ndarray = RnaForOde.sigmoide(self.S1)

        self.Y: numpy.ndarray = numpy.dot(
            self.W2.T,
            self.X1
        )

    def retropropagar(self):
        self.dY: numpy.ndarray = numpy.dot(
            self.W2.T,
            numpy.multiply(
                numpy.dot(self.W1.T, numpy.ones((1, self.R))),
                RnaForOde.derivada_sigmoide(self.S1[:,1:])
            )
        )
        self.F: numpy.ndarray = self.ode.F(self.x0[:,1:], self.Y[:,1:])
        self.c_0: float = self.Y[0,0] - self.ode.A
        self.c_r: numpy.ndarray = self.dY - self.F

        self.vector_costos: numpy.ndarray = numpy.hstack((
            [[sqrt(self.R)*self.c_0,]], self.c_r
        ))

        self.costo_total: numpy.ndarray = (1.0/(2.0*self.R))*numpy.dot(
            self.vector_costos,
            self.vector_costos.T
        )

        self.costs_collection.append(float(self.costo_total))

        self.c: numpy.ndarray = numpy.vstack((
            [[self.c_0,]], self.c_r.T
        ))

        self.dF: numpy.ndarray = self.ode.dF(self.x0[:,1:], self.Y[:,1:])

        self.Delta: numpy.ndarray = numpy.multiply(
            numpy.dot(self.W1.T, numpy.ones((1, self.R))),
            RnaForOde.derivada_sigmoide(self.S1[:,1:])
        ) - numpy.multiply(
            numpy.dot(numpy.ones((self.neuronas_ocultas, 1)), self.dF),
            self.X1[:,1:]
        )

        self.Gamma: numpy.ndarray = numpy.hstack((
            self.X1[:,0]
            .reshape(self.neuronas_ocultas, 1), (1.0/self.R)*self.Delta
        ))

        self.dC_dW2 = numpy.dot(self.Gamma, self.c)

        self.Xi: numpy.ndarray = numpy.multiply(
            numpy.dot(self.W2, numpy.ones((1, self.R))),
            numpy.multiply(
                numpy.dot(self.W1.T, self.x0[:,1:]),
                RnaForOde.segunda_derivada_sigmoide(
                    self.S1[:,1:]
                )
            ) + numpy.multiply(
                numpy.dot(
                    numpy.ones((self.neuronas_ocultas,1)),
                    numpy.ones((1, self.R)) -
                    numpy.multiply(self.dF, self.x0[:,1:])
                ),
                RnaForOde.derivada_sigmoide(self.S1[:,1:])
            )
        )

        self.Theta: numpy.ndarray = numpy.hstack((
            self.ode.x0_0*numpy.multiply(
                self.W2,
                RnaForOde.derivada_sigmoide(self.S1[:,0])
                .reshape((self.neuronas_ocultas, 1))
            ), (1.0/self.R)*self.Xi
        ))

        self.dC_dW1: numpy.ndarray = numpy.dot(self.Theta, self.c)

        self.Lambda: numpy.ndarray = numpy.multiply(
            numpy.dot(self.W2, numpy.ones((1, self.R))),
            numpy.multiply(
                -numpy.dot(self.W1.T, numpy.ones((1, self.R))),
                RnaForOde.segunda_derivada_sigmoide(
                    self.S1[:,1:]
                )
            ) + numpy.multiply(
                numpy.dot(numpy.ones((self.neuronas_ocultas, 1)), self.dF),
                RnaForOde.derivada_sigmoide(self.S1[:,1:])
            )
        )

        self.Upsilon: numpy.ndarray = numpy.hstack((
            -numpy.multiply(
                self.W2,
                RnaForOde.derivada_sigmoide(self.S1[:,0])
                .reshape((self.neuronas_ocultas, 1))
            ), (1.0/self.R)*self.Lambda
        ))

        self.dC_db1: numpy.ndarray = numpy.dot(self.Upsilon, self.c)

    def corregir(self):
        self.b1 = self.b1 - self.eta*self.dC_db1
        self.W1 = self.W1 - self.eta*self.dC_dW1.T
        self.W2 = self.W2 - self.eta*self.dC_dW2

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


if __name__ == "__main__":
    R: int = 5
    ode = Ode(
        x0_0=0.0,
        A=0.0,
        F=lambda X, Y: -Y + numpy.exp(-X),
        dF=lambda X, Y: numpy.ones((1, R))
    )
    x0: numpy.ndarray = numpy.random.rand(1, R)
    neuronas_ocultas: int = 5
    rna = RnaForOde(
        R=R,
        x0=x0,
        neuronas_ocultas=neuronas_ocultas,
        epocas=10000,
        ode=ode
    )
    for epoca in range(rna.epocas):
        rna.propagar()
        rna.retropropagar()
        rna.corregir()
    