import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from torchvision import transforms

from kaggle.datasets import PictureDataset


def get_kaggle_test_loader(batch_size=100):
    device, use_cuda = get_device()
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dataset = PictureDataset(os.path.join(parent_directory, "kaggle", "data", "testset"), transform=transforms.ToTensor())
    test_loader = DataLoader(
        test_dataset, batch_size, pin_memory=use_cuda
    )
    return test_loader


def get_kaggle_data_loaders(batch_size_train, batch_size_eval):
    device, use_cuda = get_device()

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dataset = PictureDataset(os.path.join(parent_directory, "kaggle", "data", "trainset"),transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64,(0.8,1.0)),
        transforms.ToTensor()]))

    # Shuffle the data and split it into a training and a validation set
    validation_split = 0.2
    random_seed = 42
    indices = list(range(len(train_dataset)))
    split = int(np.floor(validation_split * len(train_dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, sampler=SubsetRandomSampler(train_indices), pin_memory=use_cuda
    )
    validation_loader = DataLoader(
        train_dataset, batch_size=batch_size_eval, sampler=SubsetRandomSampler(validation_indices), pin_memory=use_cuda
    )
    test_loader = get_kaggle_test_loader(batch_size_eval)
    return train_loader, validation_loader, test_loader


def print_score(which, y_true, y_predicted, logfile):
    target_names = ["Dog", "Cat"]
    print(which + "Score:")
    print(which + "Score:", file=logfile)
    print(classification_report(y_true, y_predicted, target_names=target_names))
    print(classification_report(y_true, y_predicted, target_names=target_names), file=logfile)


def get_device():
    """
    Get information on whether cuda is supported or not on this computer

    :return: device (which type of device is supported) cuda/cpu, use_cuda (True if cuda is supported)
    """
    # If a GPU is available, use it
    # PyTorch uses an elegant way to keep the code device agnostic
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False

    return device, use_cuda
