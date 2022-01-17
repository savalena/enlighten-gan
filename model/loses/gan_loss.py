import torch
import torch.nn as nn


class GanLoss:
    def __init__(self):
        self.loss = nn.BCELoss()

    def __call__(self, input, is_real):
        if is_real:
            is_real_tensor = input.new_ones(input.size(), dtype=torch.float16, requires_grad=False)
        else:
            is_real_tensor = input.new_zeros(input.size(), dtype=torch.float16, requires_grad=False)

        return self.loss(input, is_real_tensor)
