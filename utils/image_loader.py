from PIL import Image


def load_image(path):
    img = Image.open(path).convert('RGB')
    return img
