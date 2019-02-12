import os

from torch.utils.data import Dataset
from PIL import Image
import numpy

CATEGORIES = {
    "Dog": 0,
    "Cat": 1
}


class PictureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Cats & Dogs Picture Dataset

        :param root_dir:  Absolute path to the parent directory with all the images
        """
        training_set = True if "trainset" in root_dir.lower() else False

        self.root_dir = root_dir
        self.image_names = numpy.array([])
        self.labels = numpy.array([])
        self.transform = transform

        for path, sub_directories, files in os.walk(root_dir):
            for image_name in files:
                if not ".jpg" in image_name:
                    continue
                self.image_names = numpy.append(self.image_names, image_name)
                if training_set:
                    self.labels = numpy.append(self.labels, label_string_to_id(image_name.split(".")[1]))
                else:
                    self.labels = numpy.append(self.labels, -1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, img_index):
        # Load image
        image = Image.open(os.path.join(self.root_dir, self.image_names[img_index]))

        if image.mode == "L":
            image = image.convert("RGB")

        label = self.labels[img_index]
        image_name = self.image_names[img_index]

        if self.transform:
            image = self.transform(image)
        
        return image, label, image_name


def label_string_to_id(string):
    return CATEGORIES[string.capitalize()]
