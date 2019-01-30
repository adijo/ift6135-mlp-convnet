#This one will just load an existing saved model and make predictions

import os
import time
import datetime

import torch 
import torch.nn as nn

import neuralnets as neuralnets
import utils as utils

start_file = "epoch22.pt"
num_classes = 2

def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("Indexing test examples...")
    test_loader = utils.get_kaggle_test_loader()
    best_model = neuralnets.KaggleNet(num_classes).to(device)
    best_model.load_state_dict(torch.load(start_file))
    predict_test_set_labels(best_model, test_loader, device)


#WARNING: Code duplication here. We should factorize!!!
#This could be moved to the "utils.py" file or somewhere else.


def predict_test_set_labels(model, test_loader, device):
    # Test the model. Set it into evaluation mode.
    model.eval()
    print("Predicting test set labels...")
    predictions = []
    all_image_names = []
    with torch.no_grad():
        for images, labels, image_names in test_loader:
            images = images.to(device, dtype=torch.float)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy().tolist()
            predictions += predicted
            all_image_names  += image_names

    target_names = ["Dog", "Cat"]

    predictions_file_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    predictions_file_path = os.path.join("predictions", predictions_file_prefix+".csv")
    print("Saving predictions to {}".format(predictions_file_path))
    with open(predictions_file_path, "w+") as predictions_file:
        predictions_file.write("id,label\n")
        for i in range(len(predictions)):
            predictions_file.write("{},{}\n".format(all_image_names[i].replace(".jpg",""), target_names[predictions[i]]))


if __name__ == '__main__':
    main()
