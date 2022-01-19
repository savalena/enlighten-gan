import torch
import torch.nn as nn


class GanLoss:
    def __init__(self):
        self.loss = nn.MSELoss(reduction='mean')

    def compute(self, input, is_real):
        # print(input[0, 0, :10, :10])
        if is_real:
            is_real_tensor = input.new_ones(input.size(), dtype=input.dtype, requires_grad=False,
                                            device=torch.device("cuda"))
        else:
            # is_real_tensor = input.new_zeros(input.size(), dtype=input.dtype, requires_grad=False,
            #                                  device=torch.device("cuda"))
            is_real_tensor = -input.new_ones(input.size(), dtype=input.dtype, requires_grad=False,
                           device=torch.device("cuda"))

        output = self.loss(input, is_real_tensor)
        # print(output.size())
        # print(output.item())
        return output
