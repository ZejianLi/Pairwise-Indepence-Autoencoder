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
from Models.Beta_VAE import Beta_VAE


class Improved_Beta_VAE(Beta_VAE):
    
    # ================ init part ================
    
    def __init__(self, gamma_ = 30, **kwargs): # parameters
        super().__init__(**kwargs)

        assert gamma_ > 1
        
        self.beta = None
        self.gamma_ = gamma_
        self.model_name = f'Improved_Beta_VAE_{gamma_}'
        
        self.init_net_arch()
        self.init_optimizer()

        self.traversal_code_limit = 8
        
        self.iter_count = 0
        self.C_min = 0.0
        self.C_max = 5.0
        self.num_to_max = 2500
        
        logging.debug(f'{self.model_name} initialized.')



    # ================ training part ================
    
    def stepTraining(self, batch_x):
        this_batch_size = len(batch_x)
        batch_x = batch_x.to(self.device)
        self.G.train()
        self.E.train()
        
        # update C in the objective function 
        C = self.C_max 
        if self.iter_count < self.num_to_max:
            C = self.C_min + (self.C_max - self.C_min) / self.num_to_max * self.iter_count
        
        with torch.enable_grad():
            r""" \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \gamma * \left| \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z)) - C \right|. """

            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            out = self.G(z)
            
            reconstruction_loss = F.mse_loss(out, batch_x, size_average=False ).div(this_batch_size)
            KL_loss = self._kl_divergence(mu, log_var).div(this_batch_size) 

            # modified obj func
            total_loss = reconstruction_loss + self.gamma_ * torch.abs(KL_loss - C)

            # update
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'KL_loss': KL_loss.item(),
                     'disentangled_loss': abs(KL_loss.item() - C),
                     'total_loss':total_loss.item(),
                    }

        self.iter_count += 1

        return loss_dict
