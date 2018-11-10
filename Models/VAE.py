""" 
Python 3.6 
PyTorch 0.4
"""

import logging
from Models.Beta_VAE import Beta_VAE

class VAE(Beta_VAE):
    
    # ================ init part ================
    
    def __init__(self, **kwargs): # parameters
        super().__init__(beta_ = 1, **kwargs)

        self.model_name = 'VAE'
        
        logging.debug(f'{self.model_name} initialized.')
