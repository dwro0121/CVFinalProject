import torch.nn as nn

from .backbone import Resnet_Backbone


class Centernet(nn.Module):

    def __init__(self, num_classes=80, model='resnet18',pretrain=True):
        super(Centernet, self).__init__()
        self.num_classes = num_classes
        self.backbone = Resnet_Backbone(model, pretrain)
        if model =='resnet18':
            self.decoder = Centernet_decoder(512)
        else:
            self.decoder = Centernet_decoder(2048)
        self.header = Centernet_header(num_classes=self.num_classes)

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return self.header(x)


class Centernet_decoder(nn.Module):
    def __init__(self, inplanes=2048, bn_momentum=0.1):
        super(Centernet_decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        num_layers = 3
        num_filters = [256, 128, 64]
        num_kernels = [4, 4, 4]

        layers = []

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


class Centernet_header(nn.Module):
    def __init__(self, num_classes=80):
        super(Centernet_header, self).__init__()
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.hm_header = Singler_header(64, self.num_classes)
        self.offsets_header = Singler_header(64, 2)
        self.wh_header = Singler_header(64, 2)

    def forward(self, x):
        hm = self.hm_header(x)
        hm = self.sigmoid(hm)
        offsets = self.offsets_header(x)
        wh = self.wh_header(x)
        output = {
            "hm": hm,
            "offsets": offsets,
            "wh": wh,
        }
        return output


class Singler_header(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Singler_header, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.out(x)

        return x
