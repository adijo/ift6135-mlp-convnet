from mlp.src.data import MNISTDataset, MNISTTestDataset
from mlp.src.layers import HiddenLayer, FinalLayer
from mlp.src.literals.activations import ActivationLiterals
from mlp.src.literals.initialization import InitLiterals
from mlp.src.multi_layered_perceptron import MultiLayeredPerceptron
import matplotlib.pyplot as plt
import os


def run_experiment(hyper_parameters):
    experiment = None
    use_comet = False
    if use_comet:
        import comet_ml.Experiment
        experiment = comet_ml.Experiment(api_key=os.environ["COMET_API_KEY"],
                                         project_name=os.environ["COMET_PROJECT_NAME"],
                                         workspace=os.environ["COMET_WORKSPACE"])

    for param, value in hyper_parameters.items():
        if param == "layers":
            for i, layer in enumerate(value):
                if experiment:
                    experiment.log_parameter("layer_{}_num_hidden_units".format(str(i + 1)), layer.get_num_neurons())
                    experiment.log_parameter("layer_{}_activation".format(str(i + 1)), layer.get_activation_type())
                    experiment.log_parameter("layer_{}_init".format(str(i + 1)), layer.get_init_type())
                else:
                    print("layer_{}_num_hidden_units".format(str(i + 1)), layer.get_num_neurons())
                    print("layer_{}_activation".format(str(i + 1)), layer.get_activation_type())
                    print("layer_{}_init".format(str(i + 1)), layer.get_init_type())
        else:
            if experiment:
                experiment.log_parameter(param, value)
            else:
                print(param, value)

    mnist = MNISTDataset(hyper_parameters["batch_size"])
    mnist_valid = MNISTTestDataset(100)
    mlp = MultiLayeredPerceptron(hyper_parameters["layers"])
    average_loss, valid_accuracy = mlp.train(mnist, mnist_valid, epochs=hyper_parameters["epochs"], experiment=experiment)
    return average_loss, valid_accuracy


zero_hyper_parameters = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 1000,
    "dataset": "mnist",
    "init": InitLiterals.ZERO,
    "layers": [
        HiddenLayer(784, 300, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.ZERO),
        HiddenLayer(300, 200, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.ZERO),
        FinalLayer(200, 10, activation_name=ActivationLiterals.SOFTMAX, initialization_name=InitLiterals.ZERO)
    ]
}

normal_hyper_parameters = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 1000,
    "dataset": "mnist",
    "init": InitLiterals.NORMAL,
    "layers": [
        HiddenLayer(784, 300, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.NORMAL),
        HiddenLayer(300, 200, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.NORMAL),
        FinalLayer(200, 10, activation_name=ActivationLiterals.SOFTMAX, initialization_name=InitLiterals.NORMAL)
    ]
}

glorot_hyper_parameters = {
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 1000,
    "dataset": "mnist",
    "init": InitLiterals.GLOROT,
    "layers": [
        HiddenLayer(784, 300, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        HiddenLayer(300, 200, activation_name=ActivationLiterals.RELU, initialization_name=InitLiterals.GLOROT),
        FinalLayer(200, 10, activation_name=ActivationLiterals.SOFTMAX, initialization_name=InitLiterals.GLOROT)
    ]
}

zero_loss, zero_acc = run_experiment(zero_hyper_parameters)
normal_loss, normal_acc = run_experiment(normal_hyper_parameters)
glorot_loss, glorot_acc = run_experiment(glorot_hyper_parameters)

plt.plot(range(10), zero_loss, alpha=0.6, color="#001f3f", label="Zero")
plt.plot(range(10), normal_loss, alpha=0.6, color="#85144b", label="Normal")
plt.plot(range(10), glorot_loss, alpha=0.6, color="#3D9970", label="Glorot")
plt.xlabel("Epoch")
plt.ylabel("Average loss (per epoch)")
plt.legend(loc="best")
plt.title("Loss curves for Initializations")
plt.show()


plt.plot(range(10), list(map(lambda x: 100 * x, zero_acc)), alpha=0.6, color="#001f3f", label="Zero")
plt.plot(range(10), list(map(lambda x: 100 * x, normal_acc)), alpha=0.6, color="#85144b", label="Normal")
plt.plot(range(10), list(map(lambda x: 100 * x, glorot_acc)), alpha=0.6, color="#3D9970", label="Glorot")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (per epoch)")
plt.ylim((0, 100))
plt.legend(loc="best")
plt.title("Validation Accuracy curves for Initializations")
plt.show()

