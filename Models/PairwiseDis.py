""" 
Python 3.6 
PyTorch 0.4
"""

import math
import logging
from functools import reduce

import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from ot.bregman import sinkhorn


import utils
from Models.AbstractModel import AbstractModel, Models
from Models.Beta_VAE import Beta_VAE

EPS = 1e-8

class PairwiseDis(Beta_VAE):
    """ VAE augmented with pairwise independence. """
    # ================ init part ================
    
    def __init__(self, lambda_ = 1, dist = 'MI', **kwargs): # parameters
        super().__init__(**kwargs)

        assert lambda_ >= 0
        dist = dist.upper()
        assert dist in {'MI', 'KL'}
        
        self.beta = None
        self.lambda_ = lambda_
        self.pairwise_dis = dist
        self.model_name = f'PairwiseDis_{lambda_}_{dist}'
        
        logging.debug(f'{self.model_name} initialized.')

        
    # ================ training part ================
    
    def stepTraining(self, batch_x): # test covered
        batch_x = batch_x.to(self.device)
        this_batch_size = len(batch_x) 
        self.G.train()
        self.E.train()
        
        with torch.enable_grad():
            r""" \mathbb{E}_{q_{\phi}(z \mid x )} \log p_\theta(x \mid z) 
            - \mathrm{KL}(q_{\phi}(z \mid x ) \| p(z))
            - \lambda \sum_{i \neq j} w( q_{\phi}(z_i, z_j), p(z_i)p(z_j) ) 
            # pairwise independence
            """
            
            # ================  VAE part  ================
            hidden_code = self.E(batch_x)
            mu, log_var = torch.chunk(hidden_code, 2, dim=1)  # mean and log variance.
            z = self._reparametrize(mu, log_var)
            out = self.G(z)
            reconstruction_loss = F.mse_loss(out, batch_x, reduction='sum' ).div(this_batch_size)
            disentangled_loss = self._kl_divergence(mu, log_var).div(this_batch_size)

            
            # ================  pairwise disentanglement part  ================
            total_pairwise_dist = PairwiseDis.pairwise_dist(mu, log_var, self.pairwise_dis) if self.lambda_>0 else torch.tensor(0, device=self.device)

            
            # the final loss
            total_loss = reconstruction_loss + disentangled_loss + self.lambda_ * total_pairwise_dist
            
            # update
            self.vae_optimizer.zero_grad()
            total_loss.backward()
            self.vae_optimizer.step()
            
        # losses
        loss_dict = {'reconstruction_loss': reconstruction_loss.item(), 
                     'disentangled_loss': disentangled_loss.item(),
                    f'total_pairwise_dist_{self.pairwise_dis}': total_pairwise_dist.item(),
                     'total_loss': total_loss.item(),
                    }
        
        return loss_dict
    
    
    @staticmethod
    def pairwise_dist(mu, log_var, pairwise_dis, k = 2):
        this_batch_size = len(mu)
        device = mu.device
        
        assert mu.requires_grad
        
        list_pairwise_distance = []

        # get pairwise z_i and z_j stochastically
        # shuffle mu and log_var
        idx_shuffle = torch.randperm(mu.size()[1])
        
        mu_shuffled = mu[:, idx_shuffle]
        log_var_shuffled = log_var[:, idx_shuffle]
        for mu_, log_var_ in zip( mu_shuffled.split(2, dim=1), log_var_shuffled.split(2, dim=1) ):
            assert mu_.requires_grad
            assert log_var_.requires_grad

            if mu_.size()[1]<2:
                continue

            # sampling from q_i and q_j
#             k = 2 # two copies for each q(z_i \mid x)
            samples_q_z_i_j = PairwiseDis._reparametrize(mu_, log_var_, k) # kxB * 2
            samples_p_z_i_j = torch.randn(this_batch_size * k, 2).to(device) # kxB * 2

            if pairwise_dis == 'KL':
                prob_q_zi_zj = PairwiseDis.prob_q_as_GMM(mu_, log_var_, samples_q_z_i_j)
                d = PairwiseDis.KL_normal_prior(prob_q_zi_zj, samples_q_z_i_j)
        
            elif pairwise_dis == 'MI':
                d = PairwiseDis.total_correlation_GMM(mu_, log_var_, samples_q_z_i_j)
            else:
                d = torch.zeros(1)

            list_pairwise_distance.append(d)

        # sum all the distance
        total_pairwise_dist = torch.stack(list_pairwise_distance).mean()
        
        return total_pairwise_dist
    
    
    
    @staticmethod
    def pdist(A, B): # test covered
        """ Pairwise Euclidean distance
        """
        return utils.pdist(A, B)
    
    @staticmethod
    def prob_q_as_GMM(mu, log_var, x):
        """
        The probability of x in the multiple GMMs,
        where those N Gaussians are specified by mu and log_var.
        Args:   mu - N * d
           log_var - N * d
                 x - B * d
        """
        if mu.dim()==1: mu.unsqueeze_(1)
        if log_var.dim()==1: log_var.unsqueeze_(1)
        if x.dim()==1: x.unsqueeze_(1)
        # log-probabiliy of samples of each mu and log_var 
        log_prob = utils.log_prob_of_multiple_dist(mu, log_var, x) # B * N
        # probability of q(x) as GMM
        prob_q = log_prob.exp().mean(dim=1) # B
        
        return prob_q

        
    @staticmethod
    def KL_normal_prior(prob_q, x):
        """
        The KL divergence of KL(q\|p) where p=N(0,I)
        Args:  prob_q - B
                    x - B * d
        """
        # E  log(prob_q / prob_p)
#         prob_p = utils.log_prob_standard_normal(x).exp()
#         KL = ( prob_q.div(prob_p + EPS) + EPS ).log().mean()
        prob_q += EPS
        log_prob_p = utils.log_prob_standard_normal(x)
        # E [ log_prob_q - log_prob_p ]
        KL = (prob_q.log() - log_prob_p).mean()
        
        return KL
    
    
    @staticmethod
    def total_correlation_GMM(mu, log_var, x):
        """
        Total correlation of GMM, 
        where those N Gaussians are specified by mu and log_var.
        Args:    mu - N * d
            log_var - N * d
                  x - B * d
        """
        # B
        prob_q = PairwiseDis.prob_q_as_GMM(mu, log_var, x)
        
        # B * 1
        m_l_x_each_dim = zip( mu.split(1, dim=1), log_var.split(1, dim=1), x.split(1, dim=1) )
        
#         # B
#         g_margin_prob = (PairwiseDis.prob_q_as_GMM(mu_, log_var_, x_) for mu_, log_var_, x_ in m_l_x_each_dim )
#         # B
#         product_prob_qi = reduce(torch.mul, g_margin_prob)
        # TC = E log(q \| \prod_i q_i) 
#         TC = ( prob_q.div(product_prob_qi + EPS) + EPS ).log().mean()
        
        # TC = E [ log(q) -  \sum_i log(q_i) ] 
        g_log_margin_prob = (PairwiseDis.prob_q_as_GMM(mu_, log_var_, x_).log() for mu_, log_var_, x_ in m_l_x_each_dim )
        sum_log_prob_qi = sum(g_log_margin_prob)
        log_prob_q = prob_q.log()
        TC = ( log_prob_q - sum_log_prob_qi ).mean()
        
        
#         print('------- inside =======')
#         print(prob_q)
#         print(product_prob_qi)
#         print(list(PairwiseDis.prob_q_as_GMM(mu_, log_var_, x_) for mu_, log_var_, x_ in zip( mu.split(1, dim=1), log_var.split(1, dim=1), x.split(1, dim=1) ) ))
#         print(prob_q.div(product_prob_qi + EPS) + EPS)
        
        return TC
            
            
            
            
        