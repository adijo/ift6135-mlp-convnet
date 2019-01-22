from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):

    @staticmethod
    def from_activation_name(activation_name):
        activations = {
            "relu": ReLU(),
            "softmax": Softmax()
        }

        if activation_name not in activations:
            raise Exception("Unsupported activation. %s is not supported" % activation_name)
        return activations[activation_name]

    @abstractmethod
    def apply_activation(self, x):
        pass

    @abstractmethod
    def calculate_gradient(self, x, y):
        pass


class ReLU(Activation):
    def apply_activation(self, data):
        return np.maximum(0, data)

    def calculate_gradient(self, data, target):
        return np.greater(data, 0).astype(int)


class Softmax(Activation):
    def apply_activation(self, data):
        e_x = np.exp(data - np.max(data))
        return e_x / e_x.sum(axis=0)

    def calculate_gradient(self, data, target):
        return data - target
