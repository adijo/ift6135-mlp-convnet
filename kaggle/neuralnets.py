import torch.nn as nn
import torch.nn.functional as F

"""
Assignment's limitations when it comes to the CNN structure:

"You cannot use things that we have not covered in the class, directly from the deep learning library you are
    using, such as BatchNorm/WeightNorm/LayerNorm layers, regularization techniques (including dropout), and
    optimizers such as ADAM, unless you implement them yourself"."
"You can take inspiration from some modern deep neural network architectures such as the VGG networks to improve
    the performance."
"""


class CifarNet(nn.Module):
    """
    https://arxiv.org/pdf/1412.6806.pdf

    (Attempt at 2 Pytorch implementation of the paper : STRIVING FOR SIMPLICITY - THE ALL CONVOLUTIONAL NET
    JostTobiasSpringenbergâˆ—,AlexeyDosovitskiyâˆ—,ThomasBrox,MartinRiedmiller
    Department of Computer Science - University of Freiburg - Freiburg, 79110, Germany

    Roughly corresponds to the "All-CNN-C" network.
    Batch normalisation and dropout are not part of this net since they aren't allowed for this assignment.
    Based on the code from the PyTorch Tutorial.
    TODO: Remove Dropout as it is not allowed in this assignment
    """
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Dropout(0),
            # Adapted to CIFAR here. 3 channels instead of one
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        # nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            # nn.Dropout(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            # nn.Dropout(),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        # nn.MaxPool2d(kernel_size=2, stride=2))
        # Adapted to CIFAR here. 8x8 instead of 7x7 (32x32 images instead of 28x28)
        self.fc = nn.Linear(4 * 4 * 192, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class TestNet(nn.Module):
    """
    The following CNN module follows the assignment's limitations as stated at the top of the file
    """
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
