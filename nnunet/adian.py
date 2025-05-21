import torch
import torch.nn as nn
import torch.nn.functional as F
class AdaIN_constant(nn.Module):
    def __init__(self):
        super(AdaIN_constant, self).__init__()

    def forward(self, content_feat, style_feat):
        c_mean, c_std = self.compute_mean_std(content_feat)
        s_mean, s_std = self.compute_mean_std(style_feat)
        
        normalized_feat = (content_feat - c_mean) / c_std
        return normalized_feat * s_std + s_mean
    
    def compute_mean_std(self, feat):
        mean = feat.mean(dim=[2, 3, 4], keepdim=True)
        std = feat.std(dim=[2, 3, 4], keepdim=True)
        return mean, std


class AdaIN(nn.Module):
    def __init__(self, channels, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma_fc = nn.Linear(channels, channels)
        self.beta_fc = nn.Linear(channels, channels)
        
        # Initialize the parameters to ones for gamma and zeros for beta
        nn.init.constant_(self.gamma_fc.weight, 1.0)
        nn.init.constant_(self.gamma_fc.bias, 0.0)
        nn.init.constant_(self.beta_fc.weight, 0.0)
        nn.init.constant_(self.beta_fc.bias, 0.0)

    def forward(self, content, style):
        assert content.size() == style.size(), "Content and Style must have the same size"
        
        batch_size, channels, depth, height, width = content.size()

        # Compute mean and variance for content
        c_mean = content.mean(dim=(2, 3, 4), keepdim=True)
        c_std = content.std(dim=(2, 3, 4), keepdim=True)
        
        # Compute mean and variance for style
        s_mean = style.mean(dim=(2, 3, 4), keepdim=True)
        s_std = style.std(dim=(2, 3, 4), keepdim=True)

        # Normalize content
        normalized_content = (content - c_mean) / (c_std + self.epsilon)
        
        # Flatten style statistics to pass through fully connected layers
        s_mean_flat = s_mean.view(batch_size, channels)
        s_std_flat = s_std.view(batch_size, channels)
        
        # Compute scale (gamma) and shift (beta) from style statistics
        gamma = self.gamma_fc(s_std_flat).view(batch_size, channels, 1, 1, 1)
        beta = self.beta_fc(s_mean_flat).view(batch_size, channels, 1, 1, 1)

        # Apply adaptive instance normalization
        stylized_content = normalized_content * gamma + beta

        return stylized_content


