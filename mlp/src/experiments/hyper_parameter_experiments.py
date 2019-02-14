from comet_ml import Experiment
from data import MNISTDataset, MNISTTestDataset
from layers import HiddenLayer, FinalLayer
from literals.activations import ActivationLiterals
from literals.initialization import InitLiterals
from multi_layered_perceptron import MultiLayeredPerceptron
import numpy as np
import os


def calculate_accuracy(predictions, true):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == true[i]:
            correct += 1
    return float(correct) / len(true)


def run_experiment(hyper_parameters):
    experiment = Experiment(api_key=os.environ["COMET_API_KEY"],
                            project_name=os.environ["COMET_PROJECT_NAME"], workspace=os.environ["COMET_WORKSPACE"])
    mnist = MNISTDataset(hyper_parameters["batch_size"])
    mnist_valid = MNISTTestDataset(100)
    mlp = MultiLayeredPerceptron(hyper_parameters["layers"])
    final_loss = mlp.train(mnist, mnist_valid, epochs=hyper_parameters["epochs"], experiment=experiment)

    X_train, y_train = mnist.get_all_data()
    print(X_train.shape)
    print()
    y_train_pred = mlp.predict(X_train)
    train_accuracy = calculate_accuracy(y_train_pred, np.argmax(y_train, axis=1))

    mnist_valid = MNISTTestDataset(100)
    X_test, y_test = mnist_valid.get_all_data()

    y_test_pred = mlp.predict(X_test)
    test_accuracy = calculate_accuracy(y_test_pred, np.argmax(y_test, axis=1))

    for param, value in hyper_parameters.items():
        if param == "layers":
            for i, layer in enumerate(value):
                experiment.log_parameter("layer_{}_num_hidden_units".format(str(i + 1)), layer.get_num_neurons())
                experiment.log_parameter("layer_{}_activation".format(str(i + 1)), layer.get_activation_type())
                experiment.log_parameter("layer_{}_init".format(str(i + 1)), layer.get_init_type())
        else:
            experiment.log_parameter(param, value)
    experiment.log_parameter("total_parameters", find_total_parameters(hyper_parameters["layers"]))

    experiment.log_metric("loss", final_loss)
    experiment.log_metric("train_accuracy", train_accuracy)
    experiment.log_metric("valid_accuracy", test_accuracy)


def find_total_parameters(layers):
    total_parameters = 0
    for layer in layers:
        in_dim, out_dim = layer.get_dims()
        total_parameters += (in_dim * out_dim)
        total_parameters += out_dim
    return total_parameters


hyper_parameters = {
    "learning_rate": 0.1,
    "epochs": 300,
    "batch_size": 100,
    "dataset": "mnist",
    "layers": [
        HiddenLayer(784, 32, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        HiddenLayer(32, 32, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        FinalLayer(32, 10, activation_name=ActivationLiterals.SOFTMAX, initialization_name=InitLiterals.GLOROT)
    ]
}

run_experiment(hyper_parameters)
