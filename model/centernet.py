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
    def __init__(self, inplanes=2048, bn_momentum=0.1):
        super(Centernet_decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        num_layers = 3,
        num_filters = [256, 128, 64]
        num_kernels = [4, 4, 4]

        layers = []

        # 16,16,2048 -> 32,32,256
        # 32,32,256 -> 64,64,128
        # 64,64,128 -> 128,128,64
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        self.deconv_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)

class Centernet_header(nn.module):
    def __init__(self):
        super(Centernet_decoder, self).__init__()

    def forward(self, x):
        return x