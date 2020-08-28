import torch
import torch.nn as nn
import torch.optim as optim
from torchtest import test_suite

from src.models import UNet

torch.manual_seed(0)
device = "cpu"


def test_unet():
    batch_size = 4
    X = torch.randn(batch_size, 1, 64, 64)
    y = torch.ones(batch_size, 64, 64).to(dtype=torch.int64)

    net = UNet(in_c=X.shape[1], num_classes=y.shape[2])
    opt = optim.Adam(net.parameters())

    test_suite(model=net,
               loss_fn=nn.CrossEntropyLoss(),
               optim=opt,
               batch=[X, y],
               test_vars_change=True,
               device=device)


if __name__ == '__main__':
    test_unet()
