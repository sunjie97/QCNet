import torch
import torch.nn as nn 
import torch.nn.functional as F
from .resnet import resnet50
from .vgg import vgg16


class GAModule(nn.Module):
    
    def __init__(self, in_channels, out_channels, atrous_rates=[12, 24, 36]):
        super().__init__()
        
        self.convs = nn.ModuleList([
                nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            for dilation in atrous_rates
        ])
        
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(4)
        ])
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels*4, out_channels*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels*4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def _merge(self, features):
        features = torch.cat(features, dim=1)
        weights = self.weight(features)
        features = features * (1 + weights)
        features = self.relu(features)
        
        return features
        
    def forward(self, x):
        outs = []
        skip = self.skip(x)
        
        for layer in self.convs:
            outs.append(self.relu(layer(x) + skip))
        outs.append(self.relu(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=True) + skip))
        
        for i, layer in enumerate(self.fusion):
            outs[i] = layer(outs[i])
        
        outs = self._merge(outs)
        outs = self.out(outs)
        
        return outs 


class QCModule(nn.Module):
    
    def __init__(self, skip_channels, out_channels):
        super().__init__()
        
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv_mask = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)
        self.context_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def _get_context(self, q, v):
        b, c, h, w = q.size()
        
        q = q.view(b, c, h*w).unsqueeze(1)  # (b, 1, c, h*w)
        
        v = self.conv_mask(v)  # (b, 1, h, w)
        v = v.view(b, 1, h*w)  # (b, 1, h*w)
        v = F.softmax(v, dim=2)
        v = v.unsqueeze(-1)  # (b, 1, h*w, 1)
        
        context = torch.matmul(q, v)  # (b, 1, c, 1)
        context1 = context.view(b, c, 1, 1)
        context1 = self.context_conv(context1)
    
        context_bak = context.view(b, 1, 1, c)
        context_bak = F.softmax(context_bak, dim=-1)
        q = F.softmax(q, dim=2)
        ct = torch.matmul(context_bak, q)  # (b, 1, 1, c) x (b, 1, c, h*w) = (b, 1, 1, h*w)
        ct = ct.squeeze(1).view(b, 1, h, w)
        ct = F.layer_norm(ct, [h, w])
    
        return self.relu(context1+ct)
        
    def forward(self, x, skip, scale_factor=2):
        """
        query: 经过上采样后的 x1
        dic: x2
        """

        if scale_factor != 1:
            x = self.up_conv(x)

        skip = self.skip_conv(skip)
        
        context = self._get_context(x, skip)
        context = self.refine(context)
        
        out = self.relu(x + skip + context)
        
        return out 


class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class SegModule(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [SegBlock(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(SegBlock(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
    

class Decoder(nn.Module):
    def __init__(self, in_channels, pyramid_channels=256, out_channels=128, phase='train'):
        super().__init__()
        self.phase = phase

        in_channels = in_channels[::-1]

        self.gamodule = GAModule(in_channels[0], pyramid_channels)
        self.p5 = QCModule(in_channels[0], pyramid_channels)
        self.p4 = QCModule(in_channels[1], pyramid_channels)
        self.p3 = QCModule(in_channels[2], pyramid_channels)
        self.p2 = QCModule(in_channels[3], pyramid_channels)

        self.seg_modules = nn.ModuleList([
            SegModule(pyramid_channels, out_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2, 1, 0]
        ])

    def forward(self, *features):
        c2, c3, c4, c5 = features
        p6 = self.gamodule(c5)
        p5 = self.p5(p6, c5, scale_factor=1)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        features = [seg_module(p) for seg_module, p in zip(self.seg_modules, [p5, p4, p3, p2])]
        outs = sum(features)

        if self.phase == 'train':
            return outs, features
        else:
            return outs


class SegHead(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=2):
        super().__init__() 
        
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
        )

    def forward(self, x):
        x = self.up_block(x)

        return x 


class QCNet(nn.Module):

    def __init__(self, backbone='resnet', phase='train'):
        super().__init__()
        assert backbone in ['resnet', 'vgg']

        if backbone == 'resnet':
            self.encoder = resnet50(pretrained=True if phase == 'train' else False)
            self.decoder = Decoder(in_channels=[256, 512, 1024, 2048], phase=phase)
        else:
            self.encoder = vgg16(pretrained=True if phase == 'train' else False)
            self.decoder = Decoder(in_channels=[128, 256, 512, 512], phase=phase)

        self.seg_head = SegHead(in_channels=128, out_channels=1)

        self.phase = phase 
        if phase == 'train':
            self.aux_seg_heads = nn.ModuleList([
                SegHead(in_channels=128, out_channels=1)
                for _ in range(4)
            ])
        
        self._init_weights(phase=phase)

    def forward(self, x):
        if self.phase == 'train':
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        features = self.encoder(x)

        outs, aux_features = self.decoder(*features)
        outs = self.seg_head(outs)
        for i in range(len(aux_features)):
            aux_features[i] = self.aux_seg_heads[i](aux_features[i])
        return outs, aux_features

    def _forward_test(self, x):
        features = self.encoder(x)

        outs = self.decoder(*features)
        outs = self.seg_head(outs)

        return outs

    def _init_weights(self, phase='train'):
        if phase == 'train':
            init_modules = [self.decoder, self.seg_head, self.aux_seg_heads]
        else:
            init_modules = [self.decoder, self.seg_head]

        for module in init_modules:
            for m in module.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = QCNet(phase='train')

    x = torch.randn((2, 3, 512, 512), dtype=torch.float32)

    # out = model(x)
    # print(out.shape)

    out, aux_outs = model(x)
    print(out.shape)
    for o in aux_outs:
        print('a: ', o.shape)