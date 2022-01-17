import torch
import torch.nn as nn


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute(self, vgg, img, target):
        img_vgg = self.vgg_preprocess(img)
        target_vgg = self.vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    @staticmethod
    def vgg_preprocess(img):
        (r, g, b) = torch.chunk(img, 3, dim=1)
        img = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        img = (img + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        return img
