import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class FeatureOrthogonalLoss(nn.Module):
    def __init__(self):
        super(FeatureOrthogonalLoss, self).__init__()

    def forward(self, y1, y2):
        return torch.abs(F.cosine_similarity(y1, y2).mean())


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, size_average=True, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):  # input is softmax output
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = torch.log(input)
        if self.weight is not None:
            logpt = torch.mul(logpt, self.weight.unsqueeze(0))
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Loss(nn.Module):
    def __init__(self, weights=None):
        super(Loss, self).__init__()
        self.cls_loss = FocalLoss(alpha=0.4)
        self.feature_ortho_loss = FeatureOrthogonalLoss()
        self.deep_supervision = torch.nn.NLLLoss()
        if weights:
            self.weights = weights
        else:
            self.weights = [1, 0.3, 0.1]

    def forward(self, y, f1, f2, y1, y2, label):
        return self.weights[0] * self.cls_loss(y, label).sum() + \
            self.weights[1] * self.feature_ortho_loss(f1, f2) + \
            self.weights[2] * self.deep_supervision(y1.log(), label) + \
            self.weights[2] * self.deep_supervision(y2.log(), label)


