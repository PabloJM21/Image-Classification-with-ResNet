# Now we define another model by disabling all residual connections, and plot loss and accuracy curves for training and validation set.
class PlainBlock(nn.Module):
    """
    A block similar to ResNet's residual block but without the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class PlainNet(nn.Module):
    """
    A network with architecture similar to ResNet but without residual connections.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 32
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 32, num_blocks[0])
        self.layer2 = self._make_layer(block, 64, num_blocks[1], 2)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, n_blocks, stride=1):
        layers = []
        for _ in range(n_blocks):
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            stride = 1  # Only the first block will have the stride to downsample, subsequent blocks will have a stride of 1.

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

plain_model = PlainNet(PlainBlock, [2, 2])
plain_model.to(device)
summary(plain_model, (3,32,32))
