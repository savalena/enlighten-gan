from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img


def crop_image(img_pil, power=16):
    h, w = img_pil.size
    pix_to_crop_h = h % power
    pix_to_crop_w = w % power
    # img_pil.crop((0, 0, h - pix_to_crop_h, w - pix_to_crop_w))
    return img_pil.crop((0, 0, h - pix_to_crop_h, w - pix_to_crop_w))


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
