""" 
Python 3.6 
PyTorch 0.4
"""

import logging
import itertools
from math import pi, log

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

import utils
from Models.Beta_VAE import Beta_VAE
from Models.PairwiseDis import PairwiseDis

EPS = 1e-8

class PIAE(PairwiseDis):
    
    # ================ init part ================
    
    def __init__(self, **kwargs): # parameters
        super().__init__(**kwargs)
        
        self.model_name = f'PIAE({self.pairwise_dis})_{self.lambda_}'
        self.traversal_code_limit = 10


        self.N = Normal( torch.zeros(1, device=self.device), torch.ones(1, device=self.device), )
        
        logging.debug(f'{self.model_name} initialized.')
    
    
    def stepTraining(self, batch_x):
        batch_x = batch_x.to(self.device)
        this_batch_size = len(batch_x)
        self.G.train()
        self.E.train()
        
        
        with torch.enable_grad():
            
            # ================  PIAE part  ================
            
            r"""\max \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) + \mathbb{E}_{q_{\phi}(z)} \log p(z). """
            r"""\max - mse + log = \min mse - log """
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            # log_var > - log(2*pi) - 1
            log_var = F.relu(log_var) - log(2*pi) - 1 + EPS
            # like importance-weighted auto-encoder
            k = 2
            z = self._reparametrize(mu, log_var, k = k)
            out = self.G(z)
            
            if k>1:
                batch_x = batch_x.repeat(k,1,1,1)
            
            
#             reconstruction_loss = F.mse_loss(out, batch_x, size_average=False ).div(this_batch_size)
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' ).div(this_batch_size)
            
            
            """ 
            - H(q(z), p(z)) = \mathbb{E}_{q_{\phi}(z)} \log p(z), 
            where p(z) is standard multivariate Gaussian. 
            Thus, \log p(z) = \sum_i \log p(z_i).
            """
            cross_entropy_loss = - self.N.log_prob(z).sum(1).mean()
            
#             """ \sum_i \mathbb{E}_{q_{\phi}(z_i)} \log p(z_i) when P(z) are product of P(z_i)"""
#             cross_entropy_loss = self.N.log_prob(z).mean(0).sum()
        
    
            # ================  pairwise disentanglement part  ================
            total_pairwise_dist = PairwiseDis.pairwise_dist(mu, log_var, self.pairwise_dis) if self.lambda_>0 else torch.tensor(0., device=self.device)
                
            total_loss = reconstruction_loss + cross_entropy_loss + self.lambda_ * total_pairwise_dist

            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'cross_entropy_loss': cross_entropy_loss.item(),
                     'total_loss': total_loss.item(),
                     'log_var_mean': log_var.mean().item(),
                    f'total_pairwise_dist_{self.pairwise_dis}': total_pairwise_dist.item(),
                    }
        
        return loss_dict
