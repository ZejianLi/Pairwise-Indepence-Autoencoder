""" 
Python 3.6 
PyTorch 0.4
"""

import logging
import itertools
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from Models.AbstractModel import AbstractModel, Models
from Models.Beta_VAE import Beta_VAE


class AnaVAE(Beta_VAE):
    
    # ================ init part ================
    
    def __init__(self, lambda_ = 1, **kwargs): # parameters
        super().__init__(**kwargs)

        assert lambda_ > 0
        
        self.lambda_ = lambda_
        self.beta_ = None
        self.model_name = f'AnaVAE_{lambda_}'
        
        self.traversal_code_limit = 8

        
        self.init_net_arch()
        self.init_optimizer()
        
        self.R_iter = 0
        self.R_DELAY = 100

        logging.debug(f'{self.model_name} initialized.')

    
    def init_net_arch(self, specified_net_arch = None):
        models = Models[self.dataset] if specified_net_arch == None else specified_net_arch
        
        self.net_arch = models.net_arch
        self.E = models.Encoder(self.hidden_dim * 2) # mu and sigma
        self.G = models.Generator(self.hidden_dim, self.tanh)
        self.R = models.R_of_AnaVAE(self.hidden_dim)
        
        self.name_model_dict = {'Encoder': self.E, 
                                'Decoder': self.G, 
                                'Regularizer': self.R}
        
        self.init_net_component(**self.name_model_dict)
        
            
    def init_optimizer(self):
        beta1, beta2 = 0.5, 0.99
        self.vae_optimizer = optim.Adam(itertools.chain(self.E.parameters(), 
                                                        self.G.parameters(), 
                                                        self.R.parameters()), 
                                        lr=self.lr, 
                                        betas=(beta1, beta2))
        
        utils.print_line()
        logging.debug('Use ADAM optimizers for E, G and R.')
    

    # ================ training part ================
    
    def stepTraining(self, batch_x):
        batch_x = batch_x.to(self.device)
        self.G.train()
        self.E.train()
        self.R.train()
        
        with torch.enable_grad():
            r""" \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z)) 
            - \lambda E_{c_1, c_2, r} R(r \mid G(c_1), G(c_1)) """
            
            # ============ VAE part ============
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            out = self.G(z)
            
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' )
            disentangled_loss = self._kl_divergence(mu, log_var)

            # ============ analogical training part ============
            R_loss = torch.zeros(1).to(self.device)
            if self.R_iter > self.R_DELAY:
                
                # different noise and labels
                (first_code, second_code), diff_dims_batch = self.get_codepairs_with_labels()

                # generated sample pairs
                x_diff_1 = self.G(first_code)
                x_diff_2 = self.G(second_code)

                # output representation
                ID12 = self.R(x_diff_1, x_diff_2)

                # try to find which dim is different
                R_loss = self.CE_loss(ID12, diff_dims_batch ) 

            total_loss = reconstruction_loss + disentangled_loss + self.lambda_ * R_loss

            # update
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()

        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                     'R_loss': R_loss.item(),
                     'total_loss':total_loss.item(),
                    }
        
        return loss_dict


    def get_code_pair(self, num_, diff_dim=[0]):
        first = self.get_noise(num_)
        second = first.clone()
        pivot = int(num_) // 2

        interval = random.uniform(1,2)
        
        for i1 in diff_dim:
            second[0:pivot, i1] = first[0:pivot, i1] + interval
            second[pivot:, i1]  = first[pivot:, i1] - interval

        return first, second
        

    def get_codepairs_with_labels(self):
        """ Get code pairs which are different in specific dims
        """
        with torch.no_grad():

            R_batch_size = 16 # 
            
            diff_dims_batch_list = []
            first_code_batch_list = []
            second_code_batch_list = []

            for diff_dim in range(self.hidden_dim):
                # each code pair diff in diff_dim
                first_code, second_code = self.get_code_pair(R_batch_size, diff_dim = [diff_dim])

                # in batch list
                first_code_batch_list.append(first_code)
                second_code_batch_list.append(second_code)
                diff_dims_batch_list.append(torch.ones(R_batch_size).mul(diff_dim).long().to(self.device) )
            
            # the total diff codes        
            first_code  = torch.cat(first_code_batch_list, 0)
            second_code = torch.cat(second_code_batch_list, 0)
            
            # the diff dim
            diff_dims_batch = torch.cat(diff_dims_batch_list, 0)

        return (first_code, second_code), diff_dims_batch