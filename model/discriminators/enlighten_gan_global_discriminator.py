import torch.nn as nn


class GlobalDiscriminatorEnlightenGAN(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        sequence = []
        channels = [3, 64, 128, 256, 512, 512, 512, 1]
        s = [2, 2, 2, 2, 2, 1, 1]
        for i in range(n):
            sequence += [
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=s[i], padding=2),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        sequence += [nn.Conv2d(channels[-2], channels[-1], 4, stride=s[-1], padding=2)]
        ### no activation, the same as in article; they don't use sigmoid in relativistic D
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
