import torch


class AttentionMapGray:
    def __init__(self):
        self.red_val = 0.299
        self.green_val = 0.587
        self.blue_val = 0.114

    def __call__(self, image):
        r, g, b = image[0] + 1, image[1] + 1, image[2] + 1
        gray = 1. - (self.red_val * r + self.green_val * g + self.blue_val * b) / 2.
        gray = torch.unsqueeze(gray, 0)
        return gray


class AttentionMapGrayInverse:
    def __init__(self):
        self.red_val = 0.299
        self.green_val = 0.587
        self.blue_val = 0.114

    def __call__(self, image):
        r, g, b = image[0] + 1, image[1] + 1, image[2] + 1
        gray = (self.red_val * r + self.green_val * g + self.blue_val * b) / 2.
        gray = torch.unsqueeze(gray, 0)
        return gray
