import os
import torch.tensor
from torch.utils.data import Dataset
from skimage import io


class PictureDataset(Dataset):
    def __init__(self, root_dir):
        """
        Cats & Dogs Picture Dataset

        :param root_dir:  Absolute path to the parent directory with all the images.
            Each sub-folder within will be considered a class label.
        """
        training_set = True if "trainset" in root_dir.lower() else False

        self.root_dir = root_dir
        self.images_relative_paths = torch.Tensor([])
        self.labels = torch.Tensor([])
        for path, sub_directories, files in os.walk(root_dir):
            for name in files:
                self.images_relative_paths.cat(os.path.join(path, name))
                if training_set:
                    self.labels.cat(path.split(".")[1])
                else:
                    self.labels.cat("")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, img_index):
        img_absolute_path = os.path.join(self.root_dir, self.images_relative_paths[img_index])
        image = io.imread(img_absolute_path)
        label = self.labels[img_index]

        return image, label
