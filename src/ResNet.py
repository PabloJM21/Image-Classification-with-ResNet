# We write a class ResidualBlock(nn.Module) where we define the blocks used in the ResNet
class ResidualBlock(nn.Module):
    """
    A ResNet residual block with two convolutional layers.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after the first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after the second convolution.
        downsample (nn.Module): Downsample layer if input and output dimensions differ.

    Methods:
        forward(x): Passes the input through the residual block and returns the output.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass for the ResidualBlock.

        Parameters:
            x (Tensor): The input tensor with shape (N, C, H, W).
        
        Returns:
            Tensor: Output tensor after adding the block input to the block output.
        """
        identity = x
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        
        if self.downsample:
            identity = self.downsample(identity)
            
        out = residual + identity
        out = self.relu(out)
        return out


# We write a class ResNet(nn.Module) where we define the model
class ResNet(nn.Module):
    """
    ResNet model constructor for image classification with residual blocks.

    Attributes:
        conv (nn.Conv2d): Initial convolutional layer.
        bn (nn.BatchNorm2d): Initial batch normalization layer.
        layer1 (nn.Sequential): First set of residual blocks.
        layer2 (nn.Sequential): Second set of residual blocks with increased channels and downsampling.
        avg_pool (nn.AvgPool2d): Average pooling layer.
        fc (nn.Linear): Fully connected layer for classification.

    Methods:
        _make_layer(block, out_channels, blocks, stride): Constructs a layer of residual blocks.
        forward(x): Defines the forward pass of the model.
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
        """
        Constructs a sequential layer of residual blocks.

        Parameters:
            block (nn.Module): The residual block class.
            out_channels (int): Number of output channels for the blocks.
            blocks (int): Number of residual blocks in the layer.
            stride (int): Stride for downsampling, default is 1.

        Returns:
            nn.Sequential: The constructed layer of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, n_blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the ResNet.

        Parameters:
            x (Tensor): The input tensor with shape (N, C_in, H_in, W_in).

        Returns:
            Tensor: Output tensor with shape (N, num_classes) representing class scores.
        """
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


resnet_model = ResNet(ResidualBlock, [2, 2])
resnet_model.to(device)
print(resnet_model)


summary(resnet_model, (3,32,32))


