import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        batch_size = pred.shape[0]
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + 1e-6) / (pred.sum(dim=1) + target.sum(dim=1) + 1e-6)
        return (1 - dice).mean()


class TotalLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target)
        dice = self.dice(pred, target)
        return bce + self.dice_weight * dice
