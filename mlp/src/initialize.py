from abc import ABC, abstractmethod
import numpy as np


class AbstractInitialization(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, in_dim, out_dim):
        pass


def initialize_weights(in_dim, out_dim, method_name, seed=10):
    np.random.seed(seed)
    initializations = {
        "zero": Zero(),
        "normal": Normal(),
        "glorot": Glorot()
    }

    if method_name not in initializations:
        raise Exception("Invalid Initialization. %s is not supported." % method_name)

    return initializations[method_name].initialize(in_dim, out_dim)


class Zero(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        return np.zeros(shape=(in_dim, out_dim))


class Normal(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        return np.array([[np.random.standard_normal() for _ in range(out_dim)] for _ in range(in_dim)])


class Glorot(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        pass
