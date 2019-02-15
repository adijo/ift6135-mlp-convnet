from data import MNISTDataset, MNISTTestDataset
from layers import HiddenLayer, FinalLayer
from literals.activations import ActivationLiterals
from literals.initialization import InitLiterals
from multi_layered_perceptron import MultiLayeredPerceptron
import matplotlib.pyplot as plt


def run_experiment():
    layers = [
        HiddenLayer(784, 50, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        HiddenLayer(50, 50, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        FinalLayer(50, 10, activation_name=ActivationLiterals.SOFTMAX, initialization_name=InitLiterals.GLOROT)
    ]
    mnist = MNISTDataset(1)
    mnist_full = MNISTDataset(1000)

    mnist_valid = MNISTTestDataset(100)
    mlp = MultiLayeredPerceptron(layers)
    mlp.train(mnist_full, mnist_valid, 5)
    X, y = mnist.get_next_batch()

    N_values = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
    max_diffs = []
    for n in N_values:
        max_diffs.append(mlp.gradient_check(X, y, epsilon=1.0/n))

    plt.xlabel("N")
    plt.ylabel("Max Difference: True Gradient - Finite Gradient")
    plt.title("Finite Gradient Check")
    plt.semilogx(N_values, max_diffs, color="#85144b", alpha=0.6)
    plt.show()

    plt.xlabel("Epsilon")
    plt.ylabel("Max Difference: True Gradient - Finite Gradient")
    plt.title("Finite Gradient Check")
    plt.plot(list(map(lambda x: 1.0/x, N_values)), max_diffs, color="#85144b", alpha=0.6)
    plt.show()


run_experiment()