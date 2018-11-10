""" 
Python 3.6 
PyTorch 0.4
"""

import logging
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import Networks.BasicBlocks as BasicBlocks
from Models.AbstractModel import AbstractModel, Models
from Models.Beta_VAE import Beta_VAE

class FactorVAE(AbstractModel):
    
    # ================ init part ================
    
    def __init__(self, gamma_ = 1, **kwargs): # parameters
        super().__init__(**kwargs)

        assert gamma_ > 0
        
        self.gamma_ = gamma_
        self.model_name = f'FactorVAE_{gamma_}'
        
        self.traversal_code_limit = 8

        
        self.init_net_arch()
        self.init_optimizer()

        
    def init_net_arch(self, specified_net_arch = None):
        models = Models[self.dataset] if specified_net_arch == None else specified_net_arch
        
        self.net_arch = models.net_arch
        self.E = models.Encoder(self.hidden_dim * 2) # mu and sigma
        self.G = models.Generator(self.hidden_dim, self.tanh)
        
        list_layers = [BasicBlocks.block_linear_BN_RELU(self.hidden_dim, 100, leaky = 0.2) ]
        list_layers.extend( [BasicBlocks.block_linear_BN_RELU(100, 100, leaky = 0.2) for _ in range(3)] )
        list_layers.append( nn.Linear(100, 1) )
        self.D = nn.Sequential( * list_layers )
        
        self.name_model_dict = {'Encoder':self.E, 'Decoder':self.G, 'Discriminator':self.D}
        
        self.init_net_component(**self.name_model_dict)

            
    def init_optimizer(self):
        beta1, beta2 = 0.5, 0.99
        self.vae_optimizer = optim.Adam(itertools.chain(self.E.parameters(), self.G.parameters()), 
                                        lr=self.lr, 
                                        betas=(beta1, beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(beta1, beta2))
        
        utils.print_line()
        logging.debug('Use ADAM optimizers for E, G and D.')
    

    # ================ training part ================

    
    def stepTraining(self, batch_x):
        # the FactorVAE requires two training batch, we split batch_x into two batches
        this_batch_size = len(batch_x) // 2
        batch_x, batch_x_prime = batch_x[:this_batch_size], batch_x[this_batch_size:]
        
        batch_x = batch_x.to(self.device)
        batch_x_prime = batch_x_prime.to(self.device)
        
        ones  = torch.ones(this_batch_size,  device = self.device)
        zeros = torch.zeros(this_batch_size, device = self.device)
        
        
        with torch.enable_grad():
            r""" 
            \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z))
            - \gamma \mathrm{KL}( q(z) \| \tilde{q}(z) )
            
            D(z) = \frac{ q(z) }{ q(z) + \tilde{q}(z) }
            """

            # ================ train VAE ================
            
            self.G.train()
            self.E.train()
            self.D.eval()
            
            # encode
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = Beta_VAE._reparametrize(mu, log_var)
            # decode
            out = self.G(z)
            
            # for the second training part
            z_ = z.detach()
            
            # two losses of VAE
            reconstruction_loss = F.mse_loss(out, batch_x, size_average=False ).div(this_batch_size)
            disentangled_loss = Beta_VAE._kl_divergence(mu, log_var).div(this_batch_size)
            
            # TC loss
            D_z = self.D(z).squeeze()
            # \gamma log \sigmoid( D(z) )
            TC_1 = F.binary_cross_entropy_with_logits(D_z, ones, -self.gamma_)
            # -\gamma log( 1-\sigmoid( D(z) ) )
            TC_2 = F.binary_cross_entropy_with_logits(D_z, zeros, self.gamma_)
            # \gamma log( \sigmoid( D(z) ) / ( 1-\sigmoid( D(z) ) )  )
            TC_loss = TC_1 + TC_2

            # final loss
            total_loss = reconstruction_loss + disentangled_loss + TC_loss

            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()


            # ================ train D ================
            
            self.E.eval()
            self.D.train()
            
            # encode again
            hidden_code_prime = self.E(batch_x_prime)
            mu_prime, log_var_prime = torch.chunk(hidden_code_prime, 2, dim=1)  # mean and log variance.
            z_prime = Beta_VAE._reparametrize(mu_prime, log_var_prime)
            
            # permute in each hidden dim
            z_permuted = [ z_j[ torch.randperm(this_batch_size) ] for z_j in z_prime.split(1, dim = 1) ]
            z_permuted = torch.cat(z_permuted, dim=1).detach()
            
            # discriminate
            D_z_ = self.D(z_).squeeze()
            D_z_permuted = self.D(z_permuted).squeeze()
            
            # the loss
            TC_1 = F.binary_cross_entropy_with_logits( D_z_, ones )
            TC_2 = F.binary_cross_entropy_with_logits( D_z_permuted, zeros )
            D_loss = TC_1 + TC_2
            
            # update
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
            

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                     'TC_loss': TC_loss.item(),
                     'total_loss':total_loss.item(),
                     'D_loss':D_loss.item()
                    }
        
        return loss_dict
