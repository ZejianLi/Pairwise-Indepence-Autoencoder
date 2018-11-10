import math
import functools

import torch.nn.functional as F
from torch import nn
import utils

# ================================ Network blocks part ================================

def block_conv_k3s2p1_BN_RELU(in_channel_size, out_channel_size, leaky = 0):
    """
    >>> block_conv_k3s2p1_BN_RELU(10, 13, 0.07)
    Sequential(
      (0): Conv2d(10, 13, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(13, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.07, inplace)
    )
    """
    model_list = []
    model_list.append( nn.Conv2d( in_channel_size, out_channel_size, \
        kernel_size=3, stride=2, padding=1, bias=False ) )
    model_list.append( nn.BatchNorm2d(out_channel_size) )
    model_list.append( nn.ReLU( inplace=True ) if leaky==0 else nn.LeakyReLU(leaky, inplace=True) )
    
    return nn.Sequential(*model_list)


def block_conv_k4s2p1_BN_RELU(in_channel_size, out_channel_size, leaky = 0):
    """
    >>> block_conv_k4s2p1_BN_RELU(11, 15, 0.09)
    Sequential(
      (0): Conv2d(11, 15, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(15, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.09, inplace)
    )
    """
    model_list = []
    model_list.append( nn.Conv2d( in_channel_size, out_channel_size, \
        kernel_size=4, stride=2, padding=1, bias=False ) )
    model_list.append( nn.BatchNorm2d(out_channel_size) )
    model_list.append( nn.ReLU( inplace=True ) if leaky==0 else nn.LeakyReLU(leaky, inplace=True) )
    
    return nn.Sequential(*model_list)


def block_deconv_k4s2p1_BN_RELU(in_channel_size, out_channel_size, leaky = 0):
    """
    >>> block_deconv_k4s2p1_BN_RELU(13, 17, 0.02)
    Sequential(
      (0): ConvTranspose2d(13, 17, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(17, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.02, inplace)
    )
    """
    model_list = []
    model_list.append( nn.ConvTranspose2d( in_channel_size, out_channel_size, \
        kernel_size=4, stride=2, padding=1, bias=False ) )
    model_list.append( nn.BatchNorm2d(out_channel_size) )
    model_list.append( nn.ReLU( inplace=True ) if leaky==0 else nn.LeakyReLU(leaky, inplace=True) )
    
    return nn.Sequential(*model_list)


def block_linear_BN_RELU(in_size, out_size, leaky = 0):
    """
    >>> block_linear_BN_RELU(11, 9, 0.11)
    Sequential(
      (0): Linear(in_features=11, out_features=9, bias=True)
      (1): BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.11, inplace)
    )
    >>> block_linear_BN_RELU(19, 7, 0)
    Sequential(
      (0): Linear(in_features=19, out_features=7, bias=True)
      (1): BatchNorm1d(7, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace)
    )
    """
    model_list = []
    model_list.append( nn.Linear(in_size, out_size) )
    model_list.append( nn.BatchNorm1d(out_size) )
    model_list.append( nn.ReLU( inplace=True ) if leaky==0 else nn.LeakyReLU(leaky, inplace=True) )
    
    return nn.Sequential(*model_list)


