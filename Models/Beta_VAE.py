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


class Beta_VAE(AbstractModel):
    
    # ================ init part ================
    
    def __init__(self, beta_ = 1, **kwargs): # parameters
        super().__init__(**kwargs)

        assert beta_ >= 1
        
        self.beta = beta_
        self.model_name = f'Beta_VAE_{beta_}' if beta_!=1 else 'VAE'
        
        self.traversal_code_limit = 15

        
        self.init_net_arch()
        self.init_optimizer()
        
        logging.debug(f'{self.model_name} initialized.')

    
    def init_net_arch(self, specified_net_arch = None):
        models = Models[self.dataset] if specified_net_arch == None else specified_net_arch
        
        self.net_arch = models.net_arch
        self.E = models.Encoder(self.hidden_dim * 2) # mu and sigma
        self.G = models.Generator(self.hidden_dim, self.tanh)
        
        self.name_model_dict = {'Encoder': self.E, 'Decoder': self.G}
        
        self.init_net_component(**self.name_model_dict)
        
            
    def init_optimizer(self):
        beta1, beta2 = 0.5, 0.99
        self.vae_optimizer = optim.Adam(itertools.chain(self.E.parameters(), self.G.parameters()), 
                                        lr=self.lr, 
                                        betas=(beta1, beta2))
        
        utils.print_line()
        logging.debug('Use ADAM optimizers for E and G.')
    

    # ================ training part ================
    
    @staticmethod
    def _reparametrize(mu, log_var, k = 1):
        r""" z = mean + eps * sigma where eps is sampled from \mathcal{N}(0, I). 
             Get k samples from each mu and log_var 
         """
        if k != 1:
            dim = mu.size()[1]
            mu = mu.repeat(k, 1)
            log_var = log_var.repeat(k, 1)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_var / 2)    # 2 for convert var to std
        return z


    @staticmethod
    def _kl_divergence(mu, log_var):
        return - torch.sum( 0.5 * ( 1 + log_var - mu**2 - log_var.exp() ) )
    
    
    def stepTraining(self, batch_x):
        batch_x = batch_x.to(self.device)
        self.G.train()
        self.E.train()
        
        
        with torch.enable_grad():
            r""" \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \beta * \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z)). """
            
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            out = self.G(z)
            
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' )
            disentangled_loss = self._kl_divergence(mu, log_var)

            total_loss = reconstruction_loss + self.beta * disentangled_loss

            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                     'total_loss':total_loss.item(),
                    }
        
        return loss_dict
