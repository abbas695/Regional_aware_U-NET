import torch
import torch.nn as nn
import torch.nn.functional as F

class SidewiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SidewiseConv, self).__init__()
        # Bottom-Up Convolution (K_ij)
        self.conv_bottom_up = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # Left-Right Convolution (K_ik)
        self.conv_left_right = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 3), padding=(1, 0, 1))
        # Front-Back Convolution (K_jk)
        self.conv_front_back = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        # Apply sidewise convolutions
        x_bottom_up = self.conv_bottom_up(x)
        x_left_right = self.conv_left_right(x)
        x_front_back = self.conv_front_back(x)

        # Concatenate along the channel dimension
        out = torch.cat((x_bottom_up, x_left_right, x_front_back), dim=1)
        return out

class TriCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TriCNN, self).__init__()
        # Standard Convolution
        self.std_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Sidewise Convolution
        self.sidewise_conv = SidewiseConv(in_channels, out_channels)
        
        # Pointwise Convolution (1x1x1)
        self.pointwise_conv = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)
        
        # Skip connection with pointwise convolution
        self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Standard Convolution
        x_std = self.std_conv(x)
        
        # Sidewise Convolution
        x_sidewise = self.sidewise_conv(x)
        
        # Concatenation of standard and sidewise convolutions
        x_concat = torch.cat((x_std, x_sidewise), dim=1)
        
        # Pointwise Convolution to reduce channels
        x_pointwise = self.pointwise_conv(x_concat)
        
        # Skip Connection
        x_skip = self.skip_conv(x)
        out = F.relu(x_pointwise + x_skip)
        return out

# Example usage
if __name__ == "__main__":
    # Input: Batch size of 1, 16 channels, 32x32x32 voxel grid
    x = torch.randn(1, 16, 32, 32, 32)
    model = TriCNN(in_channels=16, out_channels=32)
    
    # Forward pass
    output = model(x)
    print(output.shape)  # Should be [1, 32, 32, 32, 32]
