# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint   
import torch
from torch import Tensor

normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}


def get_norm(name, out_channels, groups=32):
    if "groupnorm" in name:
        return nn.GroupNorm(groups, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim=3, bias=False):
    conv = convolutions[f"Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    return conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]

import torch
import torch.nn as nn






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
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.scale_activation = torch.nn.Sigmoid()

        

    def forward(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        
        return scale

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(InputBlock, self).__init__()
        self.conv1 = get_conv(in_channels, out_channels, 3, 1)
        self.conv2 = get_conv(out_channels, out_channels, 3, 1)
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride)
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock1, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)
    def forward(self, x):
        x = self.conv1(x)
        return x
class ConvBlock2(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size, stride, **kwargs):
        super(ConvBlock2, self).__init__()
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size, stride, **kwargs)

    def forward(self, x):
        x = self.conv2(x)
        return x    


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock, self).__init__()
        self.conv_block = ConvBlock(out_channels + in_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, x, x_skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = torch.cat((x, x_skip), dim=1)
        #x = checkpoint(self.conv_block,x,use_reentrant=False)
        x = self.conv_block(x)
        return x

class UpsampleBlock_withguidance(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super(UpsampleBlock_withguidance, self).__init__()
        self.conv_with_guidance1 = ConvBlock1(in_channels+out_channels, out_channels, kernel_size, 1, **kwargs)
        self.conv_with_guidance2 = ConvBlock2(out_channels, out_channels, kernel_size, 1, **kwargs)


        self.SE=SqueezeExcitation(96,8)


    def forward(self, x, x_skip, guidance):
        x = nn.functional.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
  
        guidance=self.SE(guidance)
   
        x=guidance*x
        x = torch.cat((x,x_skip), dim=1)
        x = self.conv_with_guidance1(x)
      
        x = self.conv_with_guidance2(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)

    def forward(self, input_data):
        #OUT = checkpoint(self.conv,x,use_reentrant=False)
        OUT=self.conv(input_data)
        return OUT


class UNet3D(nn.Module):
    def __init__(
        self,
        kernels,
        strides,
    ):
        super(UNet3D, self).__init__()
        self.dim = 3
        self.n_class = 1
        self.deep_supervision = True
        self.norm = "instancenorm3d"
        self.filters = [64, 96, 128, 192, 256, 384][: len(strides)]

        down_block = ConvBlock
        self.input_block = InputBlock(5, self.filters[0], norm=self.norm)
        self.guidance_input_block_ed=InputBlock(2, self.filters[1], norm=self.norm)
        self.guidance_input_block_nec=InputBlock(2, self.filters[1], norm=self.norm)
        self.guidance_input_block_et=InputBlock(1, self.filters[1], norm=self.norm)
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
        )
        self.upsamples_WT = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[2:][::-1],
            out_channels=self.filters[1:-1][::-1],
            kernels=kernels[2:][::-1],
            strides=strides[2:][::-1],
        )
        self.upsamples_TC = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[2:][::-1],
            out_channels=self.filters[1:-1][::-1],
            kernels=kernels[2:][::-1],
            strides=strides[2:][::-1],
        )
        self.upsamples_ET = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[2:][::-1],
            out_channels=self.filters[1:-1][::-1],
            kernels=kernels[2:][::-1],
            strides=strides[2:][::-1],
        )              

        self.upsamples_WT_guidance = self.get_module_list(
            conv_block=UpsampleBlock_withguidance,
            in_channels=[self.filters[1]],
            out_channels=[self.filters[0]],
            kernels=kernels[1],
            strides=strides[1],
        )
        self.upsamples_TC_guidance = self.get_module_list(
            conv_block=UpsampleBlock_withguidance,
            in_channels=[self.filters[1]],
            out_channels=[self.filters[0]],
            kernels=kernels[1],
            strides=strides[1],
        )
        self.upsamples_ET_guidance = self.get_module_list(
            conv_block=UpsampleBlock_withguidance,
            in_channels=[self.filters[1]],
            out_channels=[self.filters[0]],
            kernels=kernels[1],
            strides=strides[1],
        )     

        self.output_block_wt = self.get_output_block(decoder_level=0)
        self.output_block_tc = self.get_output_block(decoder_level=0)
        self.output_block_et = self.get_output_block(decoder_level=0)
        self.deep_supervision_heads_WT = self.get_deep_supervision_heads()
        self.deep_supervision_heads_TC = self.get_deep_supervision_heads()
        self.deep_supervision_heads_ET = self.get_deep_supervision_heads()
        self.apply(self.initialize_weights)

    def forward(self, input_data):
        out = self.input_block(input_data)
        out_guidance_ed=self.guidance_input_block_ed(torch.cat((input_data[:,0:1,:,:,:], input_data[:,3:4,:,:,:]), dim=1))
      
        out_guidance_nec=self.guidance_input_block_nec(torch.cat((input_data[:,1:2,:,:,:], input_data[:,2:3,:,:,:]), dim=1))
        out_guidance_et=self.guidance_input_block_et(input_data[:,2:3,:,:,:])
        encoder_outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            encoder_outputs.append(out)
        #out=checkpoint(self.bottleneck,out,use_reentrant=False)
        out = self.bottleneck(out)
        WT_out=out
        TC_out=out
        ET_out=out
        WT_decoder_outputs = []
        TC_decoder_outputs = []
        ET_decoder_outputs = []
        for upsample, skip in zip(self.upsamples_WT, reversed(encoder_outputs)):
            #WT_out=checkpoint(upsample,WT_out,skip,use_reentrant=False)
            WT_out = upsample(WT_out, skip)
            WT_decoder_outputs.append(WT_out)
        for upsample, skip in zip(self.upsamples_TC, reversed(encoder_outputs)):
            #TC_out=checkpoint(upsample,TC_out,skip,use_reentrant=False)
            TC_out = upsample(TC_out, skip)
            TC_decoder_outputs.append(TC_out)
        for upsample, skip in zip(self.upsamples_ET, reversed(encoder_outputs)):
            #ET_out=checkpoint(upsample,ET_out,skip,use_reentrant=False)
            ET_out = upsample(ET_out, skip)
            ET_decoder_outputs.append(ET_out)
     
        #WT_out = self.upsamples_WT_guidance[0](WT_out, encoder_outputs[0], out_guidance_ed)
        WT_out=checkpoint(self.upsamples_WT_guidance[0],WT_out, encoder_outputs[0], out_guidance_ed,use_reentrant=False)
        WT_decoder_outputs.append(WT_out)
        #TC_out = self.upsamples_TC_guidance[0](TC_out, encoder_outputs[0], out_guidance_nec)
        TC_out=checkpoint(self.upsamples_TC_guidance[0],TC_out, encoder_outputs[0], out_guidance_nec,use_reentrant=False)
        TC_decoder_outputs.append(TC_out)
        #ET_out = self.upsamples_ET_guidance[0](ET_out, encoder_outputs[0], out_guidance_et)
        ET_out=checkpoint(self.upsamples_ET_guidance[0],ET_out, encoder_outputs[0], out_guidance_et,use_reentrant=False)
        ET_decoder_outputs.append(ET_out)
        
        WT_out = self.output_block_wt(WT_out)
        TC_out = self.output_block_tc(TC_out)
        ET_out = self.output_block_et(ET_out)
        out=torch.cat((WT_out,TC_out,ET_out),dim=1)
        if self.training and self.deep_supervision:
            out=[out]
            WT_out = [WT_out]
            TC_out = [TC_out]
            ET_out = [ET_out]
            i=0
            l=1
            for i, decoder_out in enumerate(WT_decoder_outputs[-3:-1][::-1]):
                WT_out.append(self.deep_supervision_heads_WT[i](decoder_out))
            for i, decoder_out in enumerate(TC_decoder_outputs[-3:-1][::-1]):
                TC_out.append(self.deep_supervision_heads_TC[i](decoder_out))
            for i, decoder_out in enumerate(ET_decoder_outputs[-3:-1][::-1]):
                ET_out.append(self.deep_supervision_heads_ET[i](decoder_out))
                i=i+1            
            while(i>=l):
                out.append(torch.cat((WT_out[l],TC_out[l],ET_out[l]),dim=1))
                l=l+1
        return out

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride, drop_block=False):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(1), self.get_output_block(2)])

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block):
        layers = []
        for in_channel, out_channel, kernel, stride in zip(in_channels, out_channels, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d", "convtranspose3d"]:
            nn.init.kaiming_normal_(module.weight,a=0.01)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)
