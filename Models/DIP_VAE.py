""" 
Python 3.6 
PyTorch 0.4
"""

import logging

import torch
import torch.nn.functional as F

import utils
from Models.Beta_VAE import Beta_VAE


class DIP_VAE(Beta_VAE):
    
    # ================ init part ================
    
    def __init__(self, lambda_ = 1, **kwargs): # parameters
        super().__init__(**kwargs)

        assert lambda_ > 0
        
        self.beta = None
        self.lambda_ = lambda_
        self.model_name = f'DIP_VAE_{lambda_}'
        self.traversal_code_limit = 8

        
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
            - \lambda (\sum_{i \neq j} cov( \mu(x) )_{ij}^2 + 10 * \sum_i ( cov( \mu(x) )_{ii} - 1)^2 )
            """
            
            # encode
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            # decode
            out = self.G(z)
            
            # two losses of vae
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' ).div(this_batch_size)
            disentangled_loss = self._kl_divergence(mu, log_var).div(this_batch_size)
            
            # the moments matching
            cov_matching_loss = utils.cov(mu.t()).triu(diagonal=1).pow(2).sum() \
                                + 10 * torch.var(mu, 0).sub(1).pow(2).sum()

            # final loss
            total_loss = reconstruction_loss + disentangled_loss + self.lambda_ * cov_matching_loss

            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                     'cov_matching_loss': cov_matching_loss.item(),
                     'total_loss':total_loss.item(),
                    }
        
        return loss_dict

    

    