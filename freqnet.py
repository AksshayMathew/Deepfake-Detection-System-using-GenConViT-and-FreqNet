# # # models/freqnet.py
# # import torch
# # import torch.nn as nn

# # class FreqNet(nn.Module):
# #     def __init__(self, config):
# #         super(FreqNet, self).__init__()
# #         self.layer = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=3, padding=1),
# #             nn.ReLU(),
# #             nn.AdaptiveAvgPool2d((1, 1)),
# #             nn.Flatten(),
# #             nn.Linear(64, config['num_classes'])  # You may want to adjust this
# #         )

# #     def forward(self, x):
# #         return self.layer(x)


# # models/freqnet.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FreqNet(nn.Module):
#     def __init__(self, config):
#         super(FreqNet, self).__init__()
#         self.latent_dims = config.get("latent_dims", 12544)
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),

#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(128, self.latent_dims)

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

# models/freqnet.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class FreqNet(nn.Module):
#     def __init__(self, config):
#         super(FreqNet, self).__init__()
#         self.latent_dims = config.get("latent_dims", 12544)

#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, 112, 112]
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> [B, 128, 56, 56]
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),

#             nn.AdaptiveAvgPool2d((1, 1))  # -> [B, 128, 1, 1]
#         )

#         self.fc = nn.Linear(128, self.latent_dims)

#     def forward(self, x):
#         x = self.features(x)         # -> [B, 128, 1, 1]
#         x = torch.flatten(x, 1)      # -> [B, 128]
#         x = self.fc(x)               # -> [B, latent_dims]
#         return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = ['FreqNet', 'freqnet']


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = conv1x1(planes, planes * self.expansion)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# class FreqNet(nn.Module):
#     def __init__(self, block=Bottleneck, layers=[3, 4], num_classes=1, zero_init_residual=False):
#         super(FreqNet, self).__init__()

#         self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)))
#         self.bias1 = nn.Parameter(torch.randn((64,)))
#         self.realconv1 = conv1x1(64, 64)
#         self.imagconv1 = conv1x1(64, 64)

#         self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)))
#         self.bias2 = nn.Parameter(torch.randn((64,)))
#         self.realconv2 = conv1x1(64, 64)
#         self.imagconv2 = conv1x1(64, 64)

#         self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)))
#         self.bias3 = nn.Parameter(torch.randn((256,)))
#         self.realconv3 = conv1x1(256, 256)
#         self.imagconv3 = conv1x1(256, 256)

#         self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)))
#         self.bias4 = nn.Parameter(torch.randn((256,)))
#         self.realconv4 = conv1x1(256, 256)
#         self.imagconv4 = conv1x1(256, 256)

#         self.inplanes = 64
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(512, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = [block(self.inplanes, planes, stride, downsample)]
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def hfreqWH(self, x, scale):
#         assert scale > 2
#         x = torch.fft.fft2(x, norm="ortho")
#         x = torch.fft.fftshift(x, dim=[-2, -1])
#         b, c, h, w = x.shape
#         x[:, :, h//2 - h//scale:h//2 + h//scale, w//2 - w//scale:w//2 + w//scale] = 0.0
#         x = torch.fft.ifftshift(x, dim=[-2, -1])
#         x = torch.fft.ifft2(x, norm="ortho")
#         return F.relu(torch.real(x), inplace=True)

#     def hfreqC(self, x, scale):
#         assert scale > 2
#         x = torch.fft.fft(x, dim=1, norm="ortho")
#         x = torch.fft.fftshift(x, dim=1)
#         b, c, h, w = x.shape
#         x[:, c//2 - c//scale:c//2 + c//scale, :, :] = 0.0
#         x = torch.fft.ifftshift(x, dim=1)
#         x = torch.fft.ifft(x, dim=1, norm="ortho")
#         return F.relu(torch.real(x), inplace=True)

#     def forward(self, x):
#         x = self.hfreqWH(x, 4)
#         x = F.conv2d(x, self.weight1, self.bias1)
#         x = F.relu(x, inplace=True)

#         x = self.hfreqC(x, 4)

#         x = torch.fft.fft2(x, norm="ortho")
#         x = torch.fft.fftshift(x, dim=[-2, -1])
#         x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
#         x = torch.fft.ifftshift(x, dim=[-2, -1])
#         x = torch.fft.ifft2(x, norm="ortho")
#         x = F.relu(torch.real(x), inplace=True)

#         x = self.hfreqWH(x, 4)
#         x = F.conv2d(x, self.weight2, self.bias2, stride=2)
#         x = F.relu(x, inplace=True)

#         x = self.hfreqC(x, 4)

#         x = torch.fft.fft2(x, norm="ortho")
#         x = torch.fft.fftshift(x, dim=[-2, -1])
#         x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
#         x = torch.fft.ifftshift(x, dim=[-2, -1])
#         x = torch.fft.ifft2(x, norm="ortho")
#         x = F.relu(torch.real(x), inplace=True)

#         x = self.maxpool(x)
#         x = self.layer1(x)

#         x = self.hfreqWH(x, 4)
#         x = F.conv2d(x, self.weight3, self.bias3)
#         x = F.relu(x, inplace=True)

#         x = torch.fft.fft2(x, norm="ortho")
#         x = torch.fft.fftshift(x, dim=[-2, -1])
#         x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
#         x = torch.fft.ifftshift(x, dim=[-2, -1])
#         x = torch.fft.ifft2(x, norm="ortho")
#         x = F.relu(torch.real(x), inplace=True)

#         x = self.hfreqWH(x, 4)
#         x = F.conv2d(x, self.weight4, self.bias4, stride=2)
#         x = F.relu(x, inplace=True)

#         x = torch.fft.fft2(x, norm="ortho")
#         x = torch.fft.fftshift(x, dim=[-2, -1])
#         x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
#         x = torch.fft.ifftshift(x, dim=[-2, -1])
#         x = torch.fft.ifft2(x, norm="ortho")
#         x = F.relu(torch.real(x), inplace=True)

#         x = self.layer2(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x


# def freqnet(**kwargs):
#     return FreqNet(**kwargs)



import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FreqNet', 'freqnet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class FreqNet(nn.Module):
    def __init__(self, config):
        super(FreqNet, self).__init__()

        block = Bottleneck
        layers = config.get("freq_layers", [3, 4])
        num_classes = config.get("num_classes", 1)
        zero_init_residual = config.get("zero_init_residual", False)

        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)))
        self.bias1 = nn.Parameter(torch.randn((64,)))
        self.realconv1 = conv1x1(64, 64)
        self.imagconv1 = conv1x1(64, 64)

        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)))
        self.bias2 = nn.Parameter(torch.randn((64,)))
        self.realconv2 = conv1x1(64, 64)
        self.imagconv2 = conv1x1(64, 64)

        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias3 = nn.Parameter(torch.randn((256,)))
        self.realconv3 = conv1x1(256, 256)
        self.imagconv3 = conv1x1(256, 256)

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias4 = nn.Parameter(torch.randn((256,)))
        self.realconv4 = conv1x1(256, 256)
        self.imagconv4 = conv1x1(256, 256)

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def hfreqWH(self, x, scale):
        assert scale > 2
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        b, c, h, w = x.shape
        x[:, :, h//2 - h//scale:h//2 + h//scale, w//2 - w//scale:w//2 + w//scale] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        return F.relu(torch.real(x), inplace=True)

    def hfreqC(self, x, scale):
        assert scale > 2
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1)
        b, c, h, w = x.shape
        x[:, c//2 - c//scale:c//2 + c//scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        return F.relu(torch.real(x), inplace=True)

    def forward(self, x):
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1)
        x = F.relu(x, inplace=True)

        x = self.hfreqC(x, 4)

        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = F.relu(torch.real(x), inplace=True)

        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2)
        x = F.relu(x, inplace=True)

        x = self.hfreqC(x, 4)

        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = F.relu(torch.real(x), inplace=True)

        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight3, self.bias3)
        x = F.relu(x, inplace=True)

        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = F.relu(torch.real(x), inplace=True)

        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2)
        x = F.relu(x, inplace=True)

        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = F.relu(torch.real(x), inplace=True)

        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def freqnet(**kwargs):
    return FreqNet(**kwargs)

