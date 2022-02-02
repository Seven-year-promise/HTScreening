import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class LinearARD(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, dim = 50):
        super(LinearARD, self).__init__()
        self.dim = dim
        self.weight = Parameter(torch.Tensor(dim, dim))
        self.bias = Parameter(torch.Tensor(dim))

        self.reset_parameters()

    def forward(self, x):
        """
        x^2 dot w^T - 2x dot (x^T * w^T) + [sum(w * z^2, axis=1)]^T + bias
        """

        x2w = torch.matmul(x ** 2, self.weight.t())
        xxw = torch.matmul(x, (self.weight * x).t())
        wx = (self.weight * (x ** 2)).sum(1, keepdim=True)
        return -0.5 * (x2w - 2*xxw + wx.t()) + self.bias

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        self.bias.data.zero_()

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)