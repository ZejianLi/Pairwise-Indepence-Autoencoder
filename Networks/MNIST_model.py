import logging

import torch
import torch.nn as nn
import Networks.BasicBlocks as BasicBlocks

NUM_HIDDEN_1 = 256
NUM_HIDDEN_2 = 128
NUM_CHANNELS_1 = 128
NUM_CHANNELS_2 = 64
LEAKY = 0.2
DROPOUT = 0.2
COLOR_CHANNELS = 1

net_arch = 'MNIST_network'

class Generator(nn.Module):
    """ Generator or Decoder of MNIST """
    def __init__(self, hidden_dim, Tanh, **kwargs):
        super().__init__()
        
        self.fc = nn.Sequential(
            BasicBlocks.block_linear_BN_RELU(hidden_dim,   NUM_HIDDEN_1, leaky = LEAKY),
            BasicBlocks.block_linear_BN_RELU(NUM_HIDDEN_1, NUM_CHANNELS_1 * 49, leaky = LEAKY)
        )
        
        # transpose convolution layers.
        self.dconv = nn.Sequential(
            BasicBlocks.block_deconv_k4s2p1_BN_RELU(NUM_CHANNELS_1, NUM_CHANNELS_2, leaky = LEAKY), 
            nn.ConvTranspose2d(in_channels = NUM_CHANNELS_2, out_channels = 1, kernel_size = 4, stride = 2, padding = 1) 
            )
        
        self.out = nn.Tanh() if Tanh else nn.Sigmoid()
        logging.debug(f'Generator out {self.out}')


    def forward(self, in_noise):
        x = self.fc(in_noise)
        x = x.view(-1, NUM_CHANNELS_1, 7, 7)
        x = self.dconv(x)
        return self.out(x)
        
        

class Discriminator(nn.Module):
    """ Discriminator for GAN"""
    def __init__(self, num_PAC = 1, **kwargs):
        super(Discriminator, self).__init__()

        # conv feature
        self.conv = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS * num_PAC, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(LEAKY),
            BasicBlocks.block_conv_k3s2p1_BN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, leaky = LEAKY),
            BasicBlocks.block_conv_k3s2p1_BN_RELU(NUM_CHANNELS_1, NUM_CHANNELS_1 * 2, leaky = LEAKY)
        )
        
        self.fc = BasicBlocks.block_linear_BN_RELU(NUM_CHANNELS_1 * 2 * 4 * 4, NUM_HIDDEN_1, leaky = LEAKY)
        
        self.last_layer = nn.Linear(NUM_HIDDEN_1, 1)
        
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 2 * 4 * 4)
        fc = self.fc(conv)
        out = self.last_layer(fc)
        return out.squeeze()



class Discriminator_InsNorm(Discriminator):
    """docstring for Discriminator_InsNorm"""
    def __init__(self, num_PAC = 1, **kwargs):
        super(Discriminator_InsNorm, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS * num_PAC, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(LEAKY),
            BasicBlocks.block_conv_k3s2p1_IN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, leaky = LEAKY),
            BasicBlocks.block_conv_k3s2p1_IN_RELU(NUM_CHANNELS_1, NUM_CHANNELS_1 * 2, leaky = LEAKY)
            )
        
        self.fc = BasicBlocks.block_linear_LN_RELU(NUM_CHANNELS_1 * 2 * 4 * 4, NUM_HIDDEN_1, leaky = LEAKY)


class Discriminator_GP(Discriminator):
    def __init__(self, num_PAC = 1, **kwargs):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS * num_PAC, NUM_CHANNELS_2, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(LEAKY),
            BasicBlocks.block_conv_k3s2p1_GN_RELU(NUM_CHANNELS_2, NUM_CHANNELS_1, leaky = LEAKY),
            BasicBlocks.block_conv_k3s2p1_GN_RELU(NUM_CHANNELS_1, NUM_CHANNELS_1 * 2, leaky = LEAKY)
            )
        
        self.fc = BasicBlocks.block_linear_LN_RELU(NUM_CHANNELS_1 * 2 * 4 * 4, NUM_HIDDEN_1, leaky = LEAKY)


class Encoder(Discriminator):
    """docstring for Encoder"""
    def __init__(self, hidden_dim, **kwargs):
        super(Encoder, self).__init__()

        self.last_layer = nn.Linear(NUM_HIDDEN_1, hidden_dim)
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 2 * 4 * 4)
        fc = self.fc(conv)
        out = self.last_layer(fc)
        return out


class R_of_AnaVAE(Discriminator):
    """ The R of AnaVAE, the same structure as Discriminator 
    but the channels of Conv are halved and with dropout
    """
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()

        # conv feature
        self.conv = nn.Sequential(
            nn.Conv2d(COLOR_CHANNELS * 2, NUM_CHANNELS_2 // 2, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(LEAKY),
            nn.Dropout2d(DROPOUT) ,
            BasicBlocks.block_conv_k3s2p1_BN_RELU(NUM_CHANNELS_2 // 2, NUM_CHANNELS_1 // 2, leaky = LEAKY),
            nn.Dropout2d(DROPOUT),
            BasicBlocks.block_conv_k3s2p1_BN_RELU(NUM_CHANNELS_1 // 2, NUM_CHANNELS_1 // 2 * 2, leaky = LEAKY),
            nn.Dropout2d(DROPOUT)
        )

        self.fc = BasicBlocks.block_linear_BN_RELU(NUM_CHANNELS_1 // 2 * 2 * 4 * 4, NUM_HIDDEN_1, leaky = LEAKY)

        self.last_layer = nn.Linear(NUM_HIDDEN_1, hidden_dim)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 // 2 * 2 * 4 * 4)
        fc = self.fc(conv)
        out = self.last_layer(fc)
        return out.squeeze()



class Discriminator_InfoGAN(Discriminator):
    """
    The Discriminator of InfoGAN, requires extra layers to regress the input codes.
    """
    def __init__(self, hidden_dim, code_dim, **kwargs):
        super().__init__()
        
        self.regress_code = nn.Sequential(
            BasicBlocks.block_linear_BN_RELU(NUM_HIDDEN_1, NUM_HIDDEN_1 // 2, leaky = LEAKY),
            nn.Linear(NUM_HIDDEN_1 // 2, code_dim)
        )
            
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1, NUM_CHANNELS_1 * 2 * 4 * 4)
        fc = self.fc(conv)
        out = self.last_layer(fc)
        regressed_code = self.regress_code(fc)
        return out.squeeze(), regressed_code
