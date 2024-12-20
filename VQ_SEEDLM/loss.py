import torch.nn as nn
import torch


class ElementwiseNormalizedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(ElementwiseNormalizedMSELoss, self).__init__()
        self.epsilon = epsilon  # 0으로 나누는 것을 방지하기 위한 작은 값

    def forward(self, y_true, y_pred):
        mse_per_element = (y_true - y_pred) ** 2 / (y_true**2 + self.epsilon)
        nmse = torch.mean(mse_per_element)
        return nmse


class NormalizedMSELoss(nn.Module):
    def __init__(self, std=None):
        super(NormalizedMSELoss, self).__init__()
        self.std = std
        self.mse_fn = nn.MSELoss()

    def forward(self, y_true, y_pred):
        # mse = torch.mean((y_true - y_pred) ** 2)
        mse = self.mse_fn(y_true, y_pred)
        if self.std is not None:
            var = self.std**2
        else:
            var = torch.mean(y_true**2)
        return mse / var


def get_loss_fn(args, std=None):
    if args.loss == "nmse":
        return NormalizedMSELoss(std)
    elif args.loss == "enmse":
        return ElementwiseNormalizedMSELoss()
