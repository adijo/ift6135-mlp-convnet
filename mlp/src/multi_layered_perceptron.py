from validate import validate_mlp_architecture
import numpy as np


class MultiLayeredPerceptron(object):
    def __init__(self, layers):
        validate_mlp_architecture(layers)
        self.layers = layers

    def fprop(self, X):
        curr_input = X
        for layer in self.layers:
            curr_input = layer.forward(curr_input)
        return curr_input

    def bprop(self, y, loss, learning_rate):
        final_layer = self.layers[-1]
        final_layer_predictions = final_layer.get_activations()
        final_layer_delta = final_layer.backward(final_layer_predictions, y)
        loss.append(final_layer.average_loss(final_layer_predictions, y))
        final_layer_weights = final_layer.get_weights()

        curr_delta = final_layer_delta
        curr_weights = final_layer_weights

        for i in range(len(self.layers) - 2, -1, -1):
            curr_delta = self.layers[i].backward(curr_weights, curr_delta)
            curr_weights = self.layers[i].get_weights()
            self.layers[i].update(learning_rate)
        return final_layer.loss(final_layer_predictions, y)

    def __reset_layers(self):
        for layer in self.layers:
            layer.reset()

    def gradient_check(self, X, y, epsilon):
        """
        It's not neat right now. 
        """
        X = X.T
        y = y.T

        assert(X.shape == (784, 1))
        assert(y.shape == (10, 1))

        first_layer = self.layers[0]
        second_layer = self.layers[1]
        output_layer = self.layers[2]

        X_1 = first_layer.forward(X)
        X_2 = second_layer.forward(X_1)
        out = output_layer.forward(X_2)

        assert(out.shape == (10, 1))

        output_delta = output_layer.backward(out, y)
        output_layer.update(0.0001)
        second_layer_delta = second_layer.backward(output_layer.get_weights(), output_delta)
        second_layer_true_grad_weight = np.matmul(second_layer_delta, X_1.T).flatten()[:10]

        second_layer_weights = second_layer.get_weights()
        second_layer_grad_check = []
        for i in range(10):
            second_layer_weights[0][i] += epsilon
            X_2_plus = second_layer.forward(X_1)
            out_plus = output_layer.forward(X_2_plus)
            loss_plus = output_layer.average_loss(out_plus, y)

            second_layer_weights[0][i] -= epsilon  # reset

            second_layer_weights[0][i] -= epsilon
            X_2_minus = second_layer.forward(X_1)
            out_minus = output_layer.forward(X_2_minus)
            loss_minus = output_layer.average_loss(out_minus, y)
            second_layer_weights[0][i] += epsilon

            second_layer_grad_check.append((loss_plus - loss_minus) / (2 * epsilon))
            second_layer_weights[0][i] += epsilon  # reset
        return np.max(np.abs(np.array(second_layer_grad_check) - second_layer_true_grad_weight))

    def train(self, train_dataset, valid_dataset, epochs, learning_rate=0.0001, experiment=None):
        loss = []
        average_loss = []
        for epoch in range(epochs):
            tot_loss = 0
            while train_dataset.has_next_batch():
                X_batch, y_batch = train_dataset.get_next_batch()
                X_batch = X_batch.T
                y_batch = y_batch.T
                self.fprop(X_batch)
                tot_loss += self.bprop(y_batch, loss, learning_rate)
            average_loss.append(tot_loss / train_dataset.get_input_size())
            print("Loss", loss[-1])
            X_train, y_train = train_dataset.get_all_data()
            X_valid, y_valid = valid_dataset.get_all_data()
            train_predictions = self.predict(X_train)
            train_accuracy = self.calculate_accuracy(train_predictions, np.argmax(y_train, axis=1))
            valid_predictions = self.predict(X_valid)
            valid_accuracy = self.calculate_accuracy(valid_predictions, np.argmax(y_valid, axis=1))
            if valid_accuracy > 0.975:
                break
            if experiment:
                experiment.log_metric("train_loss", loss[-1], step=(epoch + 1))
                experiment.log_metric("valid_accuracy", valid_accuracy, step=(epoch + 1))
                experiment.log_metric("train_accuracy", train_accuracy, step=(epoch + 1))
            train_dataset.reset()
        return average_loss

    def calculate_accuracy(self, predictions, true):
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == true[i]:
                correct += 1
        return float(correct) / len(true)

    def predict(self, data):
        predictions = self.fprop(data.T)
        labels = np.argmax(predictions, axis=0)
        return labels
