import torch.nn as nn


class Centernet(nn.Module):

    def __init__(self):
        super(Centernet, self).__init__()
        self.backbone = None
        self.decoer = None
        self.header = None

    def forward(self, x):
        return x
