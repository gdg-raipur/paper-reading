import torch
import torch.nn as nn
import torchvision

class RPN(nn.Module):
    def __init__(self, in_channels, mid_channels, num_anchors):
        super(RPN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1)
        self.conv_bbox = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        rpn_cls = self.conv_cls(x)
        rpn_bbox = self.conv_bbox(x)
        return rpn_cls, rpn_bbox

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a pre-trained backbone (e.g., VGG16)
    backbone = torchvision.models.vgg16(pretrained=True).features.to(device)
    backbone_out_channels = 512  # For VGG16, the output channels are 512

    # Create the RPN
    num_anchors = 9  # Number of anchors per location
    rpn = RPN(backbone_out_channels, 512, num_anchors).to(device)

    # Example input
    x = torch.randn(1, 3, 800, 800).to(device)

    # Forward pass
    with torch.no_grad():
        features = backbone(x)
        rpn_cls, rpn_bbox = rpn(features)

    print("RPN classification output shape:", rpn_cls.shape)
    print("RPN bounding box output shape:", rpn_bbox.shape)
