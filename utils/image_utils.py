from PIL import Image


def load_image(path):
    img = Image.open(path).convert('RGB')
    return img


def crop_image(img_pil, power=16):
    h, w = img_pil.size
    pix_to_crop_h = h % power
    pix_to_crop_w = w % power
    # img_pil.crop((0, 0, h - pix_to_crop_h, w - pix_to_crop_w))
    return img_pil.crop((0, 0, h - pix_to_crop_h, w - pix_to_crop_w))
