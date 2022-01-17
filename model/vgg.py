import torch
import torch.nn as nn


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        h = self.relu(self.conv1_1(X))
        h = self.relu(self.conv1_2(h))
        h = self.maxpool2d(h)

        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.maxpool2d(h)

        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.maxpool2d(h)

        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        conv4_3 = self.conv4_3(h)
        h = self.relu(conv4_3)

        relu5_1 = nn.ReLU(self.conv5_1(h), inplace=True)
        return relu5_1
