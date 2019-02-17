import torch.nn as nn


class KaggleNetSimple2(nn.Module):
    def __init__(self, num_classes=2):
        super(KaggleNetSimple2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 192, 1),
            nn.Sigmoid()
        )

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
    
    
class KaggleNetSimple(nn.Module):
    def __init__(self, num_classes=2):
        super(KaggleNetSimple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            nn.ReLU())
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc = nn.Sequential(nn.Linear(8 * 8 * 512, 500), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(500, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.max_pool(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.max_pool(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.max_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out
