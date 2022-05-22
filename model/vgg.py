import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1)
            # layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        outs = [] 
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                outs.append(x)

        return outs[1:]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def vgg16(pretrained=True):

    model = VGG(make_layers(vgg16_cfg))
    if pretrained:
        # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', progress=False)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg16-397923af.pth', progress=False)
        for k in list(state_dict.keys()):
            if 'classifier' in k:
                del state_dict[k]
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = vgg16(pretrained=True)
    # for k in list(model.state_dict().keys()):
    #     print(k)

    x = torch.rand((2, 3, 224, 224))
    outs = model(x)
    # print(outs.shape)
    for o in outs:
        print(o.shape)
