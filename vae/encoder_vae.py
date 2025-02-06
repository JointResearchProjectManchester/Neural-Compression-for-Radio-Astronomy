import torch
import torch.nn as nn
import torch.nn.functional as F
from residual import ResidualBlock,ResidualStack
class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1)
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens)
        
        # For mean and logvar
        self.pre_latent_conv = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=2 * latent_dim,  
            kernel_size=1,
            stride=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.residual_stack(x)
        x = self.pre_latent_conv(x)
        # Split into mean and logvar
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar
