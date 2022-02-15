import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv_block(in_channels, out_channels, size):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, size, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, size, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_channels)
    )
    return block


def conv_block(in_channels, out_channels, size):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, size, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(out_channels)
    )
    return block


class GeneratorEnlightenGAN(nn.Module):
    def __init__(self, upsampling_mode='bilinear'):
        super().__init__()
        self.attention_downsample1 = nn.MaxPool2d(2)
        self.attention_downsample2 = nn.MaxPool2d(2)
        self.attention_downsample3 = nn.MaxPool2d(2)
        self.attention_downsample4 = nn.MaxPool2d(2)


        self.conv_11 = double_conv_block(4, 32, 3)
        self.downsample_1 = nn.MaxPool2d(2)

        self.conv_12 = double_conv_block(32, 64, 3)
        self.downsample_2 = nn.MaxPool2d(2)

        self.conv_13 = double_conv_block(64, 128, 3)
        self.downsample_3 = nn.MaxPool2d(2)

        self.conv_14 = double_conv_block(128, 256, 3)
        self.downsample_4 = nn.MaxPool2d(2)

        self.conv_15 = conv_block(256, 512, 3)
        self.conv_21 = conv_block(512, 512, 3)
        # self.upsample_1 = nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=True)

        self.deconv_15 = nn.Conv2d(512, 256, 3, padding=1)

        self.conv_22 = double_conv_block(512, 256, 3)
        # self.upsample_2 = nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=True)

        self.deconv_14 = nn.Conv2d(256, 128, 3, padding=1)

        self.conv_23 = double_conv_block(256, 128, 3)
        # self.upsample_3 = nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=True)

        self.deconv_13 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv_24 = double_conv_block(128, 64, 3)
        # self.upsample_4 = nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=True)

        self.deconv_12 = nn.Conv2d(64, 32, 3, padding=1)

        self.conv_25 = conv_block(64, 32, 3)
        self.conv_26 = nn.Conv2d(32, 32, 3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_27 = nn.Conv2d(32, 3, 3, padding=1)

        self.tanh = nn.Tanh()

    def forward(self, input, attention_map):
        gray_2 = self.attention_downsample1(attention_map)
        gray_3 = self.attention_downsample2(gray_2)
        gray_4 = self.attention_downsample3(gray_3)
        gray_5 = self.attention_downsample4(gray_4)
        # print(torch.cat((input, attention_map), 1).shape)
        conv11 = self.conv_11(torch.cat((input, attention_map), 1))
        x = self.downsample_1(conv11)
        conv12 = self.conv_12(x)
        x = self.downsample_2(conv12)
        conv13 = self.conv_13(x)
        x = self.downsample_3(conv13)
        conv14 = self.conv_14(x)
        x = self.downsample_4(conv14)  #

        conv15 = self.conv_15(x)  # 20x20
        x = conv15 * gray_5
        x = F.upsample(self.conv_21(x), scale_factor=2, mode='bilinear')  # 40x40

        up1 = conv14 * gray_4  # 40x40
        up1 = torch.cat((self.deconv_15(x), up1), 1)  # deconv15 38x38
        x = F.upsample(self.conv_22(up1), scale_factor=2, mode='bilinear')

        up2 = conv13 * gray_3
        up2 = torch.cat((self.deconv_14(x), up2), 1)
        x = F.upsample(self.conv_23(up2), scale_factor=2, mode='bilinear')

        up3 = conv12 * gray_2
        up3 = torch.cat((self.deconv_13(x), up3), 1)
        x = F.upsample(self.conv_24(up3), scale_factor=2, mode='bilinear')

        up4 = conv11 * attention_map
        up4 = torch.cat((self.deconv_12(x), up4), 1)
        x = self.conv_25(up4)  #
        x = self.leaky_relu(self.conv_26(x))  # should be double conv block
        x = self.conv_27(x)

        x = x * attention_map
        x = x + input

        return self.tanh(x)
        # return x
