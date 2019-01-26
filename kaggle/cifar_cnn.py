"""
Based on the following tutorial from the web:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56
"""

import time
import datetime

import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import kaggle.neuralnets as neuralnets
import kaggle.utils as utils


def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)
    # Hyper parameters
    num_epochs = 50
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    # As in the paper
    weight_decay = 0.001

    train_loader, validation_loader, test_loader = utils.get_kaggle_data_loaders()

    model = neuralnets.KaggleNet(num_classes).to(device)

    # Logging functions. Will be used later for plotting graphs
    logfile_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')

    logfile = open("results/" + logfile_prefix + ".txt", "w+")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # TODO: Don't use Adam(assignment specifications don't allow using it)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(model, file=logfile)
    print(optimizer, file=logfile)
    print("Learning rate:", learning_rate, file=logfile)
    logfile.flush()

    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        # Reset metrics for each epoch
        scheduler.step()
        full_train_predicted = []
        full_train_labels = []
        full_validation_predicted = []
        full_validation_labels = []

        correct = 0
        total = 0
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

            if (i+1) % (batch_size/2) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/total), file=logfile)
                logfile.flush()
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy:{:.2f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(), correct/total))
        utils.print_score("Training", full_train_predicted, full_train_labels, logfile)
        # At the end of the epoch, perform validation.
        if epoch % 10 == 0:
            torch.save(model, "model.bak")
            # model.save("model.bak") #Model backup at each 10 epoch, just in case.

        with torch.no_grad():
            validation_correct = 0
            validation_total = 0
            for images, labels in validation_loader:
                validation_images = images.to(device)
                validation_labels = labels.to(device)
                validation_outputs = model(validation_images)
                _, validation_predicted = torch.max(validation_outputs.data, 1)
                validation_total += validation_labels.size(0)
                validation_correct += (validation_predicted == validation_labels).sum().item()
                full_validation_predicted += validation_predicted.cpu().numpy().tolist()
                full_validation_labels += validation_labels.cpu().numpy().tolist()
            utils.print_score("Validation", full_validation_predicted, full_validation_labels, logfile)
            print("Epoch {} validation accuracy= {:.4f}".format(epoch+1, validation_correct/validation_total))
            print("Epoch {} validation accuracy= {:.4f}".format(epoch+1, validation_correct/validation_total), file=logfile)

    # Test the model. Set it into evaluation mode.
    model.eval()

    w, h = 10, 10
    confusion_matrix = [[0 for x in range(w)] for y in range(h)]

    full_test_predicted = []
    full_test_labels = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted = predicted.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            full_test_predicted += predicted
            full_test_labels += labels

            # Builds the confusion matrix
            for prediction, target in zip(predicted, labels):
                confusion_matrix[prediction][target] += 1
        utils.print_score("Test", full_test_predicted, full_test_labels, logfile)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total), file=logfile)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    np.savetxt("results\\"+logfile_prefix+"_confusion_matrix.txt", np.matrix(confusion_matrix))

    # Just for the log
    print("Predicted (row) labels vs targets (column)", file=logfile)
    for i in range(0, 10):
        for j in range(0, 10):
            print(confusion_matrix[i][j], "\t", end='', file=logfile)
        print("\n", end="", file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()
