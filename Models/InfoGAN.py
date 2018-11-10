""" 
Python 3.6 
PyTorch 0.4
"""

import logging
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from Models.AbstractModel import AbstractModel, Models


class InfoGAN(AbstractModel):
    """ InfoGAN
        InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
    """
    
    # ================ init part ================
    
    def __init__(self, lambda_,  **kwargs): # parameters
        super().__init__(**kwargs)

        self.lambda_ = lambda_
        self.model_name = f'InfoGAN_{lambda_}'
        self.code_dim = max(self.hidden_dim // 16, 2)

        self.init_net_arch()
        self.init_optimizer()
        
        logging.debug(f'{self.model_name} initialized.')

    
    def init_net_arch(self, specified_net_arch = None):
        models = Models[self.dataset] if specified_net_arch == None else specified_net_arch
        
        self.net_arch = models.net_arch
        self.D = models.Discriminator_InfoGAN(self.hidden_dim, self.code_dim) 
        self.G = models.Generator(self.hidden_dim, self.tanh)
        
        self.name_model_dict = { 'Discriminator':self.D, 'Generator':self.G }

        self.init_net_component(**self.name_model_dict)
        
    def init_optimizer(self):
        """ initialize optimizer """
        beta1, beta2 = 0.5, 0.99
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = 0)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = 0)
        self.info_optimizer = optim.Adam(itertools.chain(self.G.parameters(), self.D.parameters()), lr=self.lr, betas=(beta1, beta2))

        utils.print_line()
        logging.debug('Use ADAM optimizers for D and G.')
        
        
    def encode(self, X):
        """ encode and decode the samples X 
        with the encoder and the decoder of the model.
        """
        self.D.eval()
        X = X.to(self.device)
        with torch.no_grad():
            _, regressed_code = self.D( X )
        return regressed_code.cpu()
    
    
    # ================ training part ================
    def stepTraining(self, batch_x):
        this_batch_size = len(batch_x)
        batch_x = batch_x.to(self.device)
        batch_ones = torch.ones(this_batch_size).to(self.device)
        batch_zeros = torch.zeros(this_batch_size).to(self.device)
        
        self.D.train()
        
        with torch.enable_grad():    
            
            # ================================== train D ==================================
            r""" E_{x \sim P_r} \log D(x) """
            
            D_real, _ = self.D(batch_x)
            # loss combines a Sigmoid layer and the BCE loss
            D_real_loss = F.binary_cross_entropy_with_logits( D_real, batch_ones)
            
            r""" E_{x \sim P_g} \log ( 1- D(x) ) """
            self.G.eval()
            z = self.get_noise(this_batch_size)
            x_fake = self.G(z).detach()
            D_fake, _ = self.D(x_fake)
            D_fake_loss = F.binary_cross_entropy_with_logits( D_fake, batch_zeros )

            D_loss = D_real_loss + D_fake_loss 

            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            # ================================== train G ==================================
            r""" E_{z \sim P_z} \log D( G(z) ) + \lambda E_{c \sim P(c), x \sim G(z,c)}[log Q(c|x)] """
            
            self.G.train()
            z = self.get_noise(this_batch_size)
            c = z[:, :self.code_dim]
            x_fake = self.G(z)
            D_fake, regressed_code = self.D(x_fake)

            G_loss = F.binary_cross_entropy_with_logits( D_fake, batch_ones )
            Info_loss = F.mse_loss(regressed_code, c)
            G_Info_loss = G_loss + self.lambda_ * Info_loss

            self.info_optimizer.zero_grad()
            G_Info_loss.backward()
            self.info_optimizer.step()

            loss_dict = {'D_real_loss': D_real_loss.item(), 
                         'D_fake_loss': D_fake_loss.item(), 
                         'D_loss': D_loss.item(), 
                         'G_loss': G_loss.item(), 
                         'Info_loss': Info_loss.item(),
                        }

        return loss_dict
    