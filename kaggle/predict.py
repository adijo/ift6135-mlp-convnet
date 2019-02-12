import os
import time
import datetime

import torch

import kaggle.neuralnets as neuralnets
import kaggle.utils as utils


def main():
    """
    Before running this, place all the .jpg images of the test set directly into kaggle/data/testset.
    This code will load an existing saved model referenced by the variable start_file and make predictions
    """
    # Configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_to_use = "best_model.bak"
    num_classes = 2

    print("Using device:", device)
    print("Indexing test examples...")
    test_loader = utils.get_kaggle_test_loader()
    best_model = neuralnets.KaggleNetSimple(num_classes).to(device)
    best_model.load_state_dict(torch.load(model_to_use))
    print("Using the model {} for predictions".format(model_to_use))
    predict_test_set_labels(best_model, test_loader, device)


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
            all_image_names += image_names

    target_names = ["Dog", "Cat"]

    predictions_file_prefix = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')
    predictions_file_path = os.path.join("predictions", predictions_file_prefix+".csv")
    print("Saving predictions to {}".format(predictions_file_path))
    with open(predictions_file_path, "w+") as predictions_file:
        predictions_file.write("id,label\n")
        for i in range(len(predictions)):
            predictions_file.write("{},{}\n".format(all_image_names[i].replace(".jpg", ""), target_names[predictions[i]]))


if __name__ == '__main__':
    main()
