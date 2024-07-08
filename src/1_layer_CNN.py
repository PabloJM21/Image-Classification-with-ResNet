# We define model
class SimpleCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2,),
            nn.ReLU()
        )

        # Dummy forward pass to determine the output size after conv layers
        # for a batch size of 1 and an image size of 32x32x3
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.data.view(1, -1).size(1)
    
        # fully-connected/linear layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 10)   # 10 classes
        )
        
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = SimpleCNN()
model.to(device)

!pip install torchsummary

from torchsummary import summary

# taking the input size as (3, 32, 32)
summary(model, (3,32,32))


