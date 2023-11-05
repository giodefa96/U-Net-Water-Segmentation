import torch
from torch import nn 

# U-net is composed of several blocks with pooling in between. Each block consists of 2 convolutions,
# therefore to not repeat code at each block we define a class that summarizes it
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128, 256, 512] ):
        super(UNET, self).__init__()
        self.downs      = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ups        = nn.ModuleList()
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool       = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Down part of U-net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Up part of U-net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # self.ups.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.ups.append(DoubleConv(feature*2, feature))
            
    def forward(self, x):
        skip_connections = []
        
        # Go down the Unet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Go through the bottleneck
        x = self.bottleneck(x)
        # Reverse the skip connections list -> first are going to be used the ones added last
        skip_connections = skip_connections[::-1]
        
        # Go up the Unet
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
#             #If original input shape not divisable by 16
#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:])

            skip_connection = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](skip_connection)
        
        # Last convolution that changes channel output size
        x = self.final_conv(x)
        return x