import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNetBase', 'make_resnet_base', 'ResNet']


'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
'''


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):#浅层网络resnet18,34由BasicBlock搭建 (con3x3,batchnorm,relu)
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: #进行欠采样，在网络的某些层会用到
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):#使用bottleneck结构可以减少参数数量，三层分别使用1*1,3*3,1*1的卷积模块
    expansion = 4           #深层网络resnet50,101,152由Bottleneck搭建

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetBase(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)#7*7大小的模块
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # _make_layer方法作用是生成多个卷积层，形成一个大的模块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.out_channels = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 

        x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
        return x


def make_resnet18_base(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNetBase(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def make_resnet34_base(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNetBase(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def make_resnet50_base(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNetBase(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def make_resnet101_base(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNetBase(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def make_resnet152_base(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNetBase(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def make_resnet_base(version, pretrained=None):
    maker = {
        'resnet18': make_resnet18_base,
        'resnet34': make_resnet34_base,
        'resnet50': make_resnet50_base,
        'resnet101': make_resnet101_base,
        'resnet152': make_resnet152_base
    }
    resnet = maker[version]()
    if pretrained is not None:
        sd = torch.load(pretrained)
        sd.pop('fc.weight')
        sd.pop('fc.bias')
        resnet.load_state_dict(sd)
    return resnet


class ResNet(nn.Module):
    
    def __init__(self, version, num_classes, pretrained=None):
        super().__init__()
        self.resnet_base = make_resnet_base(version, pretrained=pretrained)
        self.fc = nn.Linear(self.resnet_base.out_channels, num_classes)

    def forward(self, x, need_features=False):
        x = self.resnet_base(x)
        feat = x
        x = self.fc(x)
        if need_features:
            return x, feat
        else:
            return x

model_dict = {
    'resnet18': [make_resnet18_base, 512],
    'resnet34': [make_resnet34_base, 512],
    'resnet50': [make_resnet50_base, 2048],
    'resnet101': [make_resnet101_base, 2048],
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128,pretrained=None,num_class=5):
        super(SupConResNet, self).__init__()
        self.resnet_base = make_resnet_base(name, pretrained=pretrained)
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
            self.fc = nn.Linear(dim_in, num_class)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        #pre = self.fc(feat)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet18', num_classes=5):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        #Linear
        self.fc = nn.Linear(feat_dim, num_classes)
        # MLP
        # self.fc = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_dim, num_classes)
        # )


    def forward(self, features):
        return self.fc(features)