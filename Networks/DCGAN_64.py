import logging

import torch
import torch.nn as nn
import Networks.BasicBlocks as BasicBlocks

# DCGAN net arch from 
# "Unsupervised representation learning with deep convolutional generative adversarial networks" ICLR 2016

COLOUR_CHANNELS = 3
INIT_CHANNELS = 6
NUM_CHANNELS = [64, 128, 256, 512]
LEAKY = 0.2

net_arch = 'DCGAN'

class Generator(nn.Module):
    """ The DCGAN generator to generate 32 * 32 images. Four blocks.  """
    def __init__(self, hidden_dim, Tanh, **kwargs):
        super().__init__()

        # parameter
        self.hidden_dim = hidden_dim
        
        # first dconv layers
        num_channels = list( reversed(NUM_CHANNELS) ) 
        
        # b * NUM_CHANNELS[-1] * 4 * 4
        self.first_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, num_channels[0], kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d( num_channels[0] ),
            nn.ReLU( inplace = True ),
        ) 
        
        # b * NUM_CHANNELS[0] * 32 * 32
        list_layers = [ BasicBlocks.block_deconv_k4s2p1_BN_RELU(num_channels[0], num_channels[1]), 
                        BasicBlocks.block_deconv_k4s2p1_BN_RELU(num_channels[1], num_channels[2]), 
                        BasicBlocks.block_deconv_k4s2p1_BN_RELU(num_channels[2], num_channels[3]), 
                      ] 
        
        self.dconvs = nn.Sequential( *list_layers )
        
        # last dconv layer
        # b * 3 * 64 * 64 
        self.last_layer = nn.ConvTranspose2d(num_channels[-1], COLOUR_CHANNELS, kernel_size = 4, stride = 2, padding = 1, bias = False) 

        # out layer
        self.out = nn.Tanh() if Tanh else nn.Sigmoid()
        logging.debug(f'Generator out {self.out}')


    def forward(self, in_noise):
        assert in_noise.size(1) == self.hidden_dim
        in_noise = in_noise.view( in_noise.size(0), in_noise.size(1), 1 , 1 )
        x = self.first_layer(in_noise)
        x = self.dconvs(x)
        x = self.last_layer(x)
        return self.out(x)
        
        

class Discriminator(nn.Module):
    """ The DCGAN Discriminator with Batch Norm. Four layers. """
    def __init__(self, num_PAC = 1, **kwargs):
        super(Discriminator, self).__init__()

        num_channels = NUM_CHANNELS
        
        # b * NUM_CHANNELS[0] * 32 * 32
        self.first_layer = nn.Sequential( 
            nn.Conv2d(COLOUR_CHANNELS * num_PAC, num_channels[0], kernel_size = 4, stride = 2, padding = 1, bias = False), 
            nn.LeakyReLU(LEAKY, inplace = True) )
        
        # b * NUM_CHANNELS[-1] * 4 * 4
        list_layers = [ 
            BasicBlocks.block_conv_k4s2p1_BN_RELU(num_channels[0], num_channels[1], leaky = LEAKY),
            BasicBlocks.block_conv_k4s2p1_BN_RELU(num_channels[1], num_channels[2], leaky = LEAKY),
            BasicBlocks.block_conv_k4s2p1_BN_RELU(num_channels[2], num_channels[3], leaky = LEAKY),
        ] 
        self.intermediate_layer = nn.Sequential( *list_layers )
        
        # b * 1 * 1 * 1
        self.last_layer = nn.Conv2d(num_channels[-1], 1, kernel_size = 4, stride = 1, padding = 0, bias = False)

        
    def forward(self, x):
        x = self.first_layer( x )
        x = self.intermediate_layer( x )
        conv = self.last_layer( x )
        out_D = conv.squeeze()
        return out_D



class Discriminator_GP(Discriminator):
    """ The DCGAN Discriminator with Instance Norm """
    def __init__(self, **kwars):
        super().__init__(**kwars)

        num_channels = NUM_CHANNELS
        
        # b * NUM_CHANNELS[-1] * 4 * 4
        list_layers = [ BasicBlocks.block_conv_k4s2p1_IN_RELU(num_channels[i0], num_channels[i0+1], leaky = LEAKY) for i0 in range(len(num_channels)-1) ]
        self.intermediate_layer = nn.Sequential( *list_layers )



class Encoder(Discriminator):
    """ docstring for Encoder """
    def __init__(self, hidden_dim, **kwargs):
        super(Encoder, self).__init__()

        num_channels = NUM_CHANNELS
        
        # b * hidden_dim * 1 * 1
        self.last_layer = nn.Conv2d(num_channels[-1], hidden_dim, kernel_size = 4, stride = 1, padding = 0)

    def forward(self, x):
        x = self.first_layer( x )
        x = self.intermediate_layer( x )
        out_D = self.last_layer( x )
        out_D.squeeze_(3)
        out_D.squeeze_(2)
        return out_D
    

    
class R_of_AnaVAE(Discriminator):
    """ The R of AnaVAE, the same structure as Discriminator 
    but the channels of Conv are halved and with dropout
    """
    def __init__(self, hidden_dim, **kwargs):
        super(Discriminator, self).__init__()

        num_channels = list( map(lambda x: x //2, NUM_CHANNELS ) )
        
        # b * NUM_CHANNELS[0] * 32 * 32
        self.first_layer = nn.Sequential( 
            nn.Conv2d(COLOUR_CHANNELS * 2, num_channels[0], kernel_size = 3, stride = 2, padding = 1), 
            nn.LeakyReLU(LEAKY, inplace = True) )
        
        # b * NUM_CHANNELS[-1] * 4 * 4
        list_layers = [ BasicBlocks.block_conv_k3s2p1_BN_RELU(num_channels[i0], num_channels[i0+1], leaky = LEAKY) for i0 in range(len(num_channels)-1) ] 
        self.intermediate_layer = nn.Sequential( *list_layers )
        
        # b * hidden_dim * 1 * 1
        self.last_layer = nn.Conv2d(num_channels[-1], hidden_dim, kernel_size = 4, stride = 1, padding = 0)

    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.first_layer( x )
        x = self.intermediate_layer( x )
        conv = self.last_layer( x )
        out_D = conv.squeeze()
        return out_D



class Discriminator_InfoGAN(Discriminator):
    """ The Discriminator of InfoGAN, requires extra layers to regress the input codes. """
    def __init__(self, hidden_dim, code_dim, **kwargs):
        super().__init__()
        
        num_channels = NUM_CHANNELS
        
        # b * hidden_dim//2 * 1 * 1
        self.regress_code = nn.Sequential(
            nn.Conv2d(num_channels[-1], hidden_dim, kernel_size = 4, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(LEAKY, inplace = True),
            BasicBlocks.block_lambda(lambda x: x.squeeze()),
            nn.Linear(hidden_dim, code_dim)
        )
        
    def forward(self, x):
        x = self.first_layer( x )
        x = self.intermediate_layer( x )
        conv = self.last_layer( x )
        out_D = conv.squeeze()
        regressed_code = self.regress_code(x).squeeze()
        return out_D, regressed_code


