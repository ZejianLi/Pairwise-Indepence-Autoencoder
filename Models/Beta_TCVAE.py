""" 
Python 3.6 
PyTorch 0.4
"""

import logging
from math import log, pi

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal as Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MultivariateNormal

import utils
from Models.Beta_VAE import Beta_VAE

EPS = 1e-8

class Beta_TCVAE(Beta_VAE):
    """ Isolating source of disentanglement in variational autoencoders"""
    
    # ================ init part ================
    
    def __init__(self, **kwargs): # parameters
        super().__init__(**kwargs)
        
        assert self.beta > 0
        self.beta_minus_one = self.beta - 1
        
        self.model_name = f'Beta_TCVAE_{self.beta}'
        
        self.traversal_code_limit = 3


        logging.debug(f'{self.model_name} initialized.')


    # ================ training part ================
        
    
    def stepTraining(self, batch_x):
        this_batch_size = batch_x.size()[0]
        batch_x = batch_x.to(self.device)
        self.G.train()
        self.E.train()
        
        with torch.enable_grad():
            r""" 
            \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z))
            (\beta - 1) \mathrm{KL}( q(z) \| \prod_j q(z_j) )
            """
            
            # encode
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            # decode
            out = self.G(z)
            
            # two losses of VAE
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' ).div(this_batch_size)
            disentangled_loss = self._kl_divergence(mu, log_var).div(this_batch_size)
            
            # the "penalty term"
            r""" 
            \mathrm{KL}( q(z) \| \prod_j q(z_j) ) 
            = \mathbb{E}_{q(z)} \log q(z) - \sum_k \mathbb{E}_{q(z)} \log q(z_k)
            = TC1_loss - TC2_loss
            """
            TC1_loss = self._TC1(mu, log_var, z)
            TC2_loss = self._TC2(mu, log_var, z) 
            
            # \mathrm{KL}( q(z) \| \prod_j q(z_j) )
            TC_loss = TC1_loss - TC2_loss 
            
            # final loss
            total_loss = reconstruction_loss + disentangled_loss + self.beta_minus_one * TC_loss
            
            # update
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                     'TC1': TC1_loss.item(),
                     'TC2': TC2_loss.item(),
                     'TC_loss': TC_loss.item(),
                     'total_loss':total_loss.item(),
                    }
        
        return loss_dict
    
    
    
    def _TC1(self, mu, log_var, z):
        r"""
        According to (5) in the paper, for a batch of samples,
        \mathbb{E}_{q(z)} \log q(z) = 1/M \sum_i^M [ \log \sum_j q( z(x_i) \mid x_j) ] - \log(NM)
        """
        N = M = len(mu)
        prob_q_z_i = utils.log_prob_of_multiple_dist(mu, log_var, z).exp().sum(dim = 1) + EPS
        mean_log_prob_q_z_i = prob_q_z_i.log().mean()
        TC1_loss = mean_log_prob_q_z_i - log(N*N)
        
        return TC1_loss


    # TC2_loss part
    def _TC2(self, mu, log_var, z):
        r"""
        \mathbb{E}_{q(z)} \log q(z_k) 
        = \mathbb{E}_{q(z_k)} \log q(z_k) 
        = 1/M \sum_i^M [ \log \sum_j q_k( z_k(x_i) \mid x_j) ] - \log(NM)
        mu, log_var, z \in Z^{M * hidden_dim}
        k in \{1, ..., hidden_dim\}
        """
        N = M = len(mu)
        TC2_culmulative = torch.tensor(0.0, device=self.device)
        # k, columns of them
        for mu_k, log_var_k, z_k in zip(mu.split(1, dim=1), log_var.split(1, dim=1), z.split(1, dim=1)):
            q_k_z_k_x_i_x_j = utils.log_prob_of_multiple_dist(mu_k, log_var_k, z_k).exp() + EPS
            log_sum_j_q_k_z_k_x_i_x_j = q_k_z_k_x_i_x_j.sum(dim = 1).log()
            E_q_z_k_log_q_z_k = log_sum_j_q_k_z_k_x_i_x_j.mean() - log(N*M)
            TC2_culmulative += E_q_z_k_log_q_z_k

        return TC2_culmulative
    
    
    @staticmethod
    def normal_log_prob(mu, log_var, z):
        log_prob = - (z - mu)**2/(2*log_var.exp() + EPS) - 0.5 * log_var - 0.5 * log(2 * pi)
        return log_prob
    
    @staticmethod
    def multivariate_normal_log_prob(mu, log_var, z):
        log_prob = 0
        for mu_, log_var_, z_ in zip(mu, log_var, z):
            log_prob += Beta_TCVAE.normal_log_prob(mu_, log_var_, z_)
        return log_prob
    
    
    
    