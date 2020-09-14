import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import dice_coeff_vectorized


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(SoftDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        """
        Args:
            input: raw logits tensor of shape (N, C, W, H)
            target: ground truth tensor of shape (N, W, H),
                    each value is 0 <= targets[i] <= num_classes - 1
        """
        # normalize logits to get probas
        input = F.softmax(input, dim=1)
        # one-hot encode ground truth masks
        target = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2)

        return 1 - dice_coeff_vectorized(input, target, reduce_fn=torch.mean)
