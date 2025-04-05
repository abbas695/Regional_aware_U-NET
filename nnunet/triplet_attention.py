import torch
from torch import nn
import numpy as np
normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}
def get_norm(out_channels):
    name="instancenorm3d"
    return normalizations[name](out_channels, affine=True)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride)
        self.norm = get_norm(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
def get_conv(in_channels, out_channels, kernel_size, stride, dim=3, bias=False):
    conv = nn.Conv3d
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)




def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]

class ZPool(nn.Module):
    def forward(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        return torch.cat([x_mean, x_max], dim=1)

class AttentionGate3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.compress = ZPool()
        self.conv = ConvLayer(2, 1, kernel_size,1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        y = self.compress(x)
        y = self.conv(y)
        y = self.activation(y)
        return x * y
    
class TripletAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.xy = AttentionGate3D(kernel_size)
        self.xz = AttentionGate3D(kernel_size)
        self.yz = AttentionGate3D(kernel_size)

    def forward(self, x):
        b, c, x_dim, y_dim, z_dim = x.shape
        x_xy = self.xy(x.permute(0, 3, 4, 1, 2)).permute(0, 3, 4, 1, 2)  # x and y
        x_xz = self.xz(x.permute(0, 2, 4, 1, 3)).permute(0, 3, 1, 4, 2)  # x and z
        x_yz = self.yz(x.permute(0, 2, 3, 4, 1)).permute(0, 1, 4, 2, 3)  # y and z
        return 1 / 3 * (x_xy + x_xz + x_yz)

# Example usage:
# input_tensor = torch.randn(8, 16, 32, 32, 32)  # [b, c, x, y, z]
# triplet_attention_3d = TripletAttention3D()
# output_tensor = triplet_attention_3d(input_tensor)
