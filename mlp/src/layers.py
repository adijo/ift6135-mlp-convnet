import numpy as np
from mlp.src.activations import Activation
from mlp.src.initialize import initialize_weights


class AbstractLayer(object):
    def __init__(self, in_dim, out_dim, activation_name, initialization_name):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_name = initialization_name
        self.activation_fn = Activation.from_activation_name(activation_name)
        self.weights = initialize_weights(in_dim, out_dim, initialization_name)
        self.activation_name = activation_name
        self.bias = np.zeros(out_dim).reshape(-1, 1)  # Biases are typically initialized to zero.

        self.prev_layer_activation = None
        self.pre_activation = None
        self.activation = None
        self.delta = None

    def forward(self, data):
        batch_size = data.shape[1]
        assert(data.shape[0] == self.in_dim)
        self.prev_layer_activation = data
        pre_activation = np.matmul(self.weights, data) + self.bias

        assert(pre_activation.shape[0] == self.out_dim)
        assert(pre_activation.shape[1] == batch_size)

        activation = self.activation_fn.apply_activation(pre_activation)

        assert(activation.shape == (self.out_dim, batch_size))

        self.pre_activation = pre_activation
        self.activation = activation
        return activation

    def _update_weight(self, learning_rate):
        # grad_weight is a function of delta
        # n_i = number of neurons in layer i
        # delta is of dimensions (n_i * batch_size)
        # prev_layer_activation is of dimensions (n_{i-1} * batch_size)
        # weight is of dimensions (n_i * n_{i-1})
        # therefore grad_weight will be dimensions (n_i * batch_size x batch_size * n_{i-1} => n_i * n_{i-1})
        assert(self.delta.shape[0] == self.out_dim)
        assert(self.prev_layer_activation.shape[0] == self.in_dim)
        grad_weight = np.matmul(self.delta, self.prev_layer_activation.T)
        assert(grad_weight.shape == (self.out_dim, self.in_dim))
        self.weights = self.weights - (learning_rate * grad_weight)

    def _update_bias(self, learning_rate):
        # Bias is a function of delta
        # Delta is of size (num_features, batch_size)
        # To find the bias we take the mean of the columns.
        assert(self.delta.shape[0] == self.out_dim)
        grad_bias = np.mean(self.delta, axis=1).reshape(-1, 1)
        assert(grad_bias.shape[0] == self.out_dim)
        self.bias = self.bias - (learning_rate * grad_bias)
        assert(self.bias.shape == (self.out_dim, 1))

    def update(self, learning_rate):
        self._update_weight(learning_rate)
        self._update_bias(learning_rate)

    def get_activations(self):
        return self.activation

    def get_weights(self):
        return self.weights

    def get_num_neurons(self):
        return self.out_dim

    def get_activation_type(self):
        return self.activation_name.value

    def get_init_type(self):
        return self.init_name.value

    def get_dims(self):
        return self.in_dim, self.out_dim

    def reset(self):
        self.delta = None
        self.activation = None
        self.pre_activation = None
        self.prev_layer_activation = None


class HiddenLayer(AbstractLayer):
    def __init__(self, in_dim, out_dim, activation_name, initialization_name):
        super().__init__(in_dim, out_dim, activation_name, initialization_name)

    def backward(self, next_layer_weights, next_layer_delta):
        # next_layer_weights is of dims: n_i+1 * n_i
        # next_layer_delta is of dims: n_i+1 * batch_size
        # pre_activation is of size: n_i * batch_size
        batch_size = self.activation.shape[1]

        assert(next_layer_weights.shape[1] == self.out_dim)
        assert(next_layer_delta.shape[1] == batch_size)
        self.delta = np.matmul(next_layer_weights.T, next_layer_delta) * \
            self.activation_fn.calculate_gradient(self.pre_activation, None)
        assert(self.delta.shape == (self.out_dim, batch_size))
        return self.delta


class FinalLayer(AbstractLayer):
    def __init__(self, in_dim, out_dim, activation_name, initialization_name):
        super().__init__(in_dim, out_dim, activation_name, initialization_name)

    def average_loss(self, predictions, target):
        delta = 1e-9
        # Predictions is of size (num_classes, batch_size)
        # Target is of size (num_classes, batch_size)
        batch_size = predictions.shape[1]
        loss = np.sum(-(target * np.log(predictions + delta))) / batch_size
        return loss

    def loss(self, predictions, target):
        delta = 1e-9
        # Predictions is of size (num_classes, batch_size)
        # Target is of size (num_classes, batch_size)
        loss = np.sum(-(target * np.log(predictions + delta)))
        return loss

    def backward(self, predictions, target):
        self.delta = self.activation_fn.calculate_gradient(predictions, target)
        return self.delta
