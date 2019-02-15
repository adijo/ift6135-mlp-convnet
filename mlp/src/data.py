from abc import ABC
from mlp.dataset import mnist
from sklearn.datasets import make_moons, make_blobs
import numpy as np
import pickle
import gzip

class AbstractDataset(ABC):
    def __init__(self, X_train, y_train, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.current_index = 0
        self.X_train = X_train
        self.y_train = y_train

    def has_next_batch(self):
        return self.current_index < len(self.X_train)

    def get_next_batch(self):
        X_batch = self.X_train[self.current_index: self.current_index + self.batch_size, :]
        y_batch = self.y_train[self.current_index: self.current_index + self.batch_size]
        self.current_index += self.batch_size
        return X_batch, y_batch

    def get_input_dimension(self):
        return self.X_train.shape[1]

    def get_target_dimension(self):
        return self.y_train.shape[1]

    def get_input_size(self):
        return self.X_train.shape[0]

    def get_all_data(self):
        return self.X_train, self.y_train

    def __len__(self):
        return self.X_train.shape[0]

    def reset(self):
        self.current_index = 0


class MNISTDataset(AbstractDataset):
    def __init__(self, batch_size):
        X_train, y_train, X_test, y_test = mnist.load()
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        super().__init__(X_train, one_hot(y_train, len(set(y_train))), batch_size)

# class MNISTTestDataset(AbstractDataset):
#     def __init__(self, batch_size):
#         with gzip.open('/Users/adityajoshi/Downloads/mnist.pkl.gz', 'rb') as mnist_gzip:
#             mnist_pickled = pickle._Unpickler(mnist_gzip)
#             mnist_pickled.encoding = 'latin1'
#             train_split, validation_split, test_split = mnist_pickled.load()
#
#         X_valid = validation_split[0].astype(np.float32) / 255.0
#         y_valid = validation_split[1].astype(np.float32)
#         super().__init__(X_valid, one_hot(y_valid, len(set(y_valid))), batch_size)


# class MNISTDataset(AbstractDataset):
#     def __init__(self, batch_size):
#         with gzip.open('/Users/adityajoshi/Downloads/mnist.pkl.gz', 'rb') as mnist_gzip:
#             mnist_pickled = pickle._Unpickler(mnist_gzip)
#             mnist_pickled.encoding = 'latin1'
#             train_split, validation_split, test_split = mnist_pickled.load()
#         X_train = train_split[0] / 255.0
#
#         y_train = train_split[1]
#         print("SHAPE", X_train[0], one_hot(y_train, len(set(y_train)))[0])
#         super().__init__(X_train, one_hot(y_train, len(set(y_train))), batch_size)
#

class MNISTTestDataset(AbstractDataset):
    def __init__(self, batch_size):
        X_train, y_train, X_test, y_test = mnist.load()
        X_test = X_test.astype(np.float32) / 255.0
        super().__init__(X_test, one_hot(y_test, len(set(y_train))), batch_size)


class MoonDataset(AbstractDataset):
    def __init__(self, batch_size, n_samples=100):
        X, y = make_moons(n_samples)
        super().__init__(X, one_hot(y, len(set(y))), batch_size)


class BlobDataset(AbstractDataset):
    def __init__(self, batch_size=100, n_samples=100):
        X, y = make_blobs(n_samples=n_samples, centers=10, n_features=10)
        super().__init__(X, one_hot(y, len(set(y))), batch_size)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
