#We define model 2
class DeepCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv. layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Conv. layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,),
            
            # Conv. layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        
            # Conv. Layer 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=16)
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
#             nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(flattened_size, 10)   # 10 classes
        )
        
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x



model2 = DeepCNN()
model2.to(device)
summary(model2, (3, 32, 32))
