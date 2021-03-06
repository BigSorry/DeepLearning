import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

class VGG(nn.Module):
    def __init__(self, features, dimension, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        nextDimension = int(dimension / 4)
        self.classifier = nn.Sequential(
            nn.Linear(dimension*7, nextDimension),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nextDimension, nextDimension),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nextDimension, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if torch.cuda.is_available():
                m = m.cuda()
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)