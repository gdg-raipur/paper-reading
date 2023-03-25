import torch.nn as nn

from resnetblock import ResNetBlock

class Resnet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            ResNetBlock(64, 128, stride=2, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )),
            ResNetBlock(128, 128),
            ResNetBlock(128, 128),
            ResNetBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            ResNetBlock(128, 256, stride=2, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256),
            ResNetBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            ResNetBlock(256, 512, stride=2, downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
            )),
            ResNetBlock(512, 512),
            ResNetBlock(512, 512)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
