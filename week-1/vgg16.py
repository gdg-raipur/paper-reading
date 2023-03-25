import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self,
                 num_classes=1000):
        super(VGG16, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # input: 224x224x3 output: 112x112x64

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # input: 112x112x64 output: 56x56x128

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # input: 56x56x128 output: 28x28x256

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # input: 28x28x256 output: 14x14x512

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        # input: 14x14x512 output: 7x7x512

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes))
        # input: 7x7x512 output: 1000

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
