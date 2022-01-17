from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Normalize, Compose


def get_transforms(train, random_cropping_size):
    tfs = []
    if train:
        tfs.append(RandomCrop(random_cropping_size))
        tfs.append(RandomHorizontalFlip())
    tfs += [ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return Compose(tfs)
