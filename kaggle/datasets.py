import os

import torch
import tensorflow
from torch.utils.data import Dataset
from skimage import io
import numpy

CATEGORIES = {
    "Dog": 0,
    "Cat": 1
}


class PictureDataset(Dataset):
    def __init__(self, root_dir):
        """
        Cats & Dogs Picture Dataset

        :param root_dir:  Absolute path to the parent directory with all the images.
            Each sub-folder within will be considered a class label.
        """
        training_set = True if "trainset" in root_dir.lower() else False

        self.root_dir = root_dir
        self.images_paths = numpy.array([])
        self.labels = numpy.array([])
        for path, sub_directories, files in os.walk(root_dir):
            for name in files:
                image_path = os.path.join(path, name)
                self.images_paths = numpy.append(self.images_paths, image_path)
                if training_set:
                    self.labels = numpy.append(self.labels, label_string_to_id(image_path.split(".")[1]))
                else:
                    self.labels = numpy.append(self.labels, -1)
                return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, img_index):
        image = io.imread(self.images_paths[img_index])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = self.labels[img_index]

        return image, label


def label_string_to_id(string):
    return CATEGORIES[string.capitalize()]
