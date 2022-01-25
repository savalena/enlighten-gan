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

        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2d_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.relu11 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu1(self.conv1_1(x))
        h = self.relu2(self.conv1_2(h))
        h = self.maxpool2d_1(h)

        h = self.relu3(self.conv2_1(h))
        h = self.relu4(self.conv2_2(h))
        h = self.maxpool2d_2(h)

        h = self.relu5(self.conv3_1(h))
        h = self.relu6(self.conv3_2(h))
        h = self.relu7(self.conv3_3(h))
        h = self.maxpool2d_3(h)

        h = self.relu8(self.conv4_1(h))
        h = self.relu9(self.conv4_2(h))
        conv4_3 = self.conv4_3(h)
        h = self.relu10(conv4_3)

        relu5_1 = self.relu11(self.conv5_1(h))
        return relu5_1
