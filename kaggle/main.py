"""
Based on the following tutorial from the web:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
"""

import os
import time
import datetime

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

import kaggle.neuralnets as neuralnets
import kaggle.utils as utils


def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)
    # Hyper parameters
    num_epochs = 50
    num_classes = 2
    batch_size = 64
    learning_rate = 0.001

    print("Indexing training and test examples...")
    train_loader, validation_loader, test_loader = utils.get_kaggle_data_loaders()

    model = neuralnets.KaggleNet(num_classes).to(device)

    # Logging functions. Will be used later for plotting graphs
    logfile_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    logfile_path = "results/" + logfile_prefix + ".txt"

    print("Creating a file at {} to track results".format(logfile_path))
    logfile = open(logfile_path, "w+")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(model, file=logfile)
    print(optimizer, file=logfile)
    print("Learning rate:", learning_rate, file=logfile)
    logfile.flush()

    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    print("Training...")
    total_step = len(train_loader)
    best_validation_accuracy = 0.0
    best_model_path = "best_model.bak"
    for epoch in range(num_epochs):
        train(model, device, total_step, scheduler, train_loader, validation_loader, criterion, optimizer, batch_size, logfile, epoch, num_epochs)
        validation_accuracy = validate(model, validation_loader, device, logfile)

        print("Epoch {} validation accuracy= {:.4f}".format(epoch + 1, validation_accuracy))
        print("Epoch {} validation accuracy= {:.4f}".format(epoch + 1, validation_accuracy), file=logfile)

        # We preserve the best performing model. Early stopping ideology
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), best_model_path)

    print("Training complete.")
    print("The model that scored the highest validation accuracy {}% was preserved and will be used for predictions.".format(best_validation_accuracy*100))

    # Clearing training model from memory
    del model
    torch.cuda.empty_cache()

    best_model = neuralnets.KaggleNet(num_classes).to(device)
    best_model.load_state_dict(torch.load(best_model_path))

    predict_test_set_labels(best_model, test_loader, device)


def train(model, device, total_step, scheduler, train_loader, validation_loader, criterion, optimizer, batch_size, logfile, epoch, num_epochs):
    # Reset metrics for each epoch
    scheduler.step()
    full_train_predicted = []
    full_train_labels = []

    correct = 0.0
    total = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        full_train_predicted += predicted.cpu().numpy().tolist()
        full_train_labels += labels.cpu().numpy().tolist()

        correct += (predicted == labels).sum().item()

        if (i + 1) % (batch_size / 2) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), correct / total), file=logfile)
            logfile.flush()
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), correct / total))

    utils.print_score("Training", full_train_labels, full_train_predicted, logfile)


def validate(model, validation_loader, device, logfile):
    full_validation_predicted = []
    full_validation_labels = []

    with torch.no_grad():
        validation_correct = 0.0
        validation_total = 0.0
        for images, labels in validation_loader:
            validation_images = images.to(device, dtype=torch.float)
            validation_labels = labels.to(device, dtype=torch.long)

            validation_outputs = model(validation_images)
            _, validation_predicted = torch.max(validation_outputs.data, 1)
            validation_total += validation_labels.size(0)
            validation_correct += (validation_predicted == validation_labels).sum().item()
            full_validation_predicted += validation_predicted.cpu().numpy().tolist()
            full_validation_labels += validation_labels.cpu().numpy().tolist()

        utils.print_score("Validation", full_validation_predicted, full_validation_labels, logfile)

    return validation_correct / validation_total


def predict_test_set_labels(model, test_loader, device):
    # Test the model. Set it into evaluation mode.
    model.eval()

    print("Predicting test set labels...")
    predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy().tolist()
            predictions += predicted

    target_names = ["Dog", "Cat"]

    predictions_file_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    predictions_file_path = os.path.join("predictions", predictions_file_prefix+".csv")
    print("Saving predictions to {}".format(predictions_file_path))
    with open(predictions_file_path, "w+") as predictions_file:
        predictions_file.write("id,label\n")
        for i in range(len(predictions)):
            predictions_file.write("{},{}\n".format(str(i), target_names[predictions[i]]))


if __name__ == '__main__':
    main()
