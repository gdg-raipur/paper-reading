from torch import nn


class YOLOv1(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self._initialize_weights()

    def backbone(self):
        # VGG 19 backbone for YOLO

        backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),  # 448x448x3 -> 224x224x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224x64 -> 112x112x64
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),  # 112x112x64 -> 112x112x128
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112x128 -> 56x56x128
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=1, padding=1
            ),  # 56x56x128 -> 56x56x256
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),  # 56x56x256 -> 56x56x256
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=1
            ),  # 56x56x256 -> 56x56x256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 56x56x256 -> 56x56x512
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56x512 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=1
            ),  # 28x28x512 -> 28x28x256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28x28x256 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=1
            ),  # 28x28x512 -> 28x28x256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28x28x256 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=1
            ),  # 28x28x512 -> 28x28x256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28x28x256 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=1, padding=1
            ),  # 28x28x512 -> 28x28x256
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),  # 28x28x256 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=1, padding=1
            ),  # 28x28x512 -> 28x28x512
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 28x28x512 -> 28x28x1024
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28x1024 -> 14x14x1024
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=1
            ),  # 14x14x1024 -> 14x14x512
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 14x14x512 -> 14x14x1024
            nn.Conv2d(
                in_channels=1024, out_channels=512, kernel_size=1, padding=1
            ),  # 14x14x1024 -> 14x14x512
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=3, padding=1
            ),  # 14x14x512 -> 14x14x1024
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 14x14x1024 -> 14x14x1024
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1
            ),  # 14x14x1024 -> 7x7x1024
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 7x7x1024 -> 7x7x1024
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=3, padding=1
            ),  # 7x7x1024 -> 7x7x1024
        )
        return backbone

    def head(self):
        head = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(4096, 7 * 7 * (5 * 2 + self.num_classes)),
            nn.Sigmoid(),
        )
        return head

    def forward(self, x):
        x = self.backbone()(x)
        x = x.view(x.size(0), -1)
        x = self.head()(x)
        x = x.view(-1, 7, 7, (5 * 2 + self.num_classes))
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
