import random


def create_patch(image, num_patches=5, patch_size=32):
    w, h = image.size(3), image.size(2)
    patches = []
    for i in range(num_patches):
        w_offset = random.randint(0, max(0, w - patch_size - 1))
        h_offset = random.randint(0, max(0, h - patch_size - 1))
        patches.append(
            image[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
        )
    return patches
