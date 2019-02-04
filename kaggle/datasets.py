import os

from torch.utils.data import Dataset
from skimage import io
from skimage import color
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

        :param root_dir:  Absolute path to the parent directory with all the images.
            Each sub-folder within will be considered a class label.
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

        #On linux, this will only work if all the files (#.Cat.jpg and #.Dog.jpg are in the same folder.)
        #image = io.imread(os.path.join(self.root_dir, self.image_names[img_index]))

        image = Image.open(os.path.join(self.root_dir, self.image_names[img_index]))
        # Convert from gray scale (1 channel) to RGB (3 channels) if needed
        #if len(image.shape) == 2:
        #    image = color.gray2rgb(image)
        if image.mode == "L":
            image = image.convert("RGB")
        # Swap the color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        
        label = self.labels[img_index]
        image_name = self.image_names[img_index]

        if self.transform:
            image=self.transform(image)
        
        return image, label, image_name


def label_string_to_id(string):
    return CATEGORIES[string.capitalize()]
