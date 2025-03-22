import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import tarfile

from six.moves import cPickle
from six.moves import urllib
from six.moves import xrange

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import combinations, product

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

import residual_stack

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_hiddens // 2,
                               kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hiddens // 2)
        
        self.conv2 = nn.Conv2d(in_channels=num_hiddens // 2, out_channels=num_hiddens,
                               kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hiddens)
        
        self.residual_stack = residual_stack.ResidualStack(num_hiddens,
                                                           num_residual_layers,
                                                           num_residual_hiddens)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BatchNorm + ReLU
        x = self.bn2(self.conv2(x))          # Conv2 + BatchNorm (no activation)
        x = self.residual_stack(x)           # Residual stack
        return x


