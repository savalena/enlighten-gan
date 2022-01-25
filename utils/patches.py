import random


def create_patch(image1, image2, image3, num_patches=5, patch_size=32):
    w, h = image1.size(3), image1.size(2)
    patches1 = []
    patches2 = []
    patches3 = []
    for i in range(num_patches):
        w_offset = random.randint(0, max(0, w - patch_size - 1))
        h_offset = random.randint(0, max(0, h - patch_size - 1))
        patches1.append(
            image1[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
        )
        patches2.append(
            image2[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
        )
        patches3.append(
            image3[:, :, h_offset:h_offset + patch_size, w_offset:w_offset + patch_size]
        )
    return patches1, patches2, patches3
