from torch import nn
import torch
class CustomLoss():
    def __init__(self):
        pass

    def __call__(self, predicted):
        abs_values = torch.abs(predicted)
        invalid_abs_values_mask = abs_values > 1
        loss = torch.mean(abs_values * invalid_abs_values_mask)
        return loss