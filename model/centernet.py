import torch.nn as nn


class Centernet(nn.Module):

    def __init__(self):
        super(Centernet, self).__init__()
        self.backbone = None
        self.decoder = Centernet_decoder()
        self.header = Centernet_header()

    def forward(self, x):
        return x

class Centernet_decoder(nn.module):
    def __init__(self):
        super(Centernet_decoder, self).__init__()

    def forward(self, x):
        return x

class Centernet_header(nn.module):
    def __init__(self):
        super(Centernet_decoder, self).__init__()

    def forward(self, x):
        return x