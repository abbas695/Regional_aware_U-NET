import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
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

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d(1)
        self.fc1 = ConvLayer(input_channels, squeeze_channels, 3, 1)
        self.fc2 = ConvLayer(squeeze_channels, input_channels, 3, 1)
        self.scale_activation = torch.nn.Sigmoid()

        

    def forward(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        self.scale_activation(scale)
        
        return scale*input

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv1 = ConvLayer(2, 1, 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        print(avg_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM3D(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM3D, self).__init__()
        self.channel_attention = SqueezeExcitation(in_planes, ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        x_out = self.spatial_attention(x_out) * x_out
        return x_out