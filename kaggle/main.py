import os
import torch
import torch.nn as nn
import torch.optim as optim
import kaggle.utils as utils
from kaggle.neuralnets import TestNet


def main():
    learning_rate = 0.005
    num_epochs = 10
    results_file_name = "_".join(["TestNet", str(learning_rate), str(num_epochs)])
    test_data_set_results_save_path = os.path.join("predictions", results_file_name)

    train_loader, validation_loader, test_loader = utils.get_data_loaders()

    device, use_cuda = utils.get_device()
    model = TestNet().to(device)
    # Note: We are not allowed to use optimizers such as Adam, according to the assignment's specifications
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Surrogate loss used for training
    train_loss_fn = nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(num_epochs):
        train(model, train_loader, train_loss_fn, optimizer, epoch, device)
        loss, accuracy = test(model, validation_loader, test_loss_fn, device)

    predict(model, test_loader, test_data_set_results_save_path)


def train(
    model,
    train_loader,
    train_loss_fn,
    optimizer,
    epoch,
    device
):
    """
    Perform one epoch of training.

    :param model:
    :param train_loader:
    :param train_loss_fn:
    :param optimizer:
    :param epoch:
    :param device:
    :return:
    """
    # Put the model into training mode
    model.train()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        # Let them code what's here
        optimizer.zero_grad()
        output = model(inputs)
        loss = train_loss_fn(output, target)
        loss.backward()
        optimizer.step()
        ###

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader) * len(inputs),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(
    model,
    test_loader,
    test_loss_fn,
    device
):
    """
    Evaluate the model on the test_loader

    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    """"""
    # Put the model into evaluation mode
    model.eval()

    test_loss = 0
    sum_correct_predictions = 0.0
    test_size = 0
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs)
            test_size += len(inputs)
            test_loss += test_loss_fn(output, target).item()  # sum up batch loss
            prediction = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            sum_correct_predictions += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= test_size
    accuracy = sum_correct_predictions / test_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, sum_correct_predictions, test_size,
        100. * accuracy))

    return test_loss, accuracy


def predict(
    model,
    test_loader,
    save_file_path
):
    """
    Predict the class labels for the the given data in test_loader. Save the results to the file save_file_path.

    :param model:
    :param test_loader:
    :param save_file_path:
    :return:
    """
    # TODO: Implement
    pass


if __name__ == '__main__':
    main()
