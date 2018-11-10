""" 
Python 3.6 
PyTorch 0.4
"""

from abc import ABC, abstractmethod
from functools import partialmethod
import logging
import os

import torch

import utils
import Networks.MNIST_model as MNIST_model
import Networks.DCGAN_64 as DCGAN_64


Models = {'mnist': MNIST_model,  
          'fashion_mnist': MNIST_model,
          'small-mnist':MNIST_model,
          'cifar10': DCGAN_64,
          }

    
class AbstractModel(ABC):
    """ the abstract class of each model """

    def __init__(self, dataset, hidden_dim, tanh = True, gpu_mode = True, lr = 1e-4, **kwargs):
        super(AbstractModel, self).__init__()

        # parameter 
        self.dataset = dataset.lower()
        self.hidden_dim = hidden_dim
        self.tanh = tanh
        self.lr = lr
        self.model_name = None
        
        self.traversal_code_limit = 6

        # device
        self.device = torch.device('cuda:0') if gpu_mode else torch.device('cpu')

        self.num_visual_samples = 6**2
        self.fixed_noise = self.get_noise( self.num_visual_samples )
        
        logging.debug('AbstractModel initialized.')


    @abstractmethod
    def init_net_arch(self, specified_net_arch = None):
        raise NotImplementedError
        
#         models = Models[self.dataset] if specified_net_arch == None else specified_net_arch
#         # your G E D R or something
        
#         self.name_model_dict = {'Encoder': self.E, 'Decoder': self.G}
        
#         self.init_net_component(**self.name_model_dict)
    
    
    def init_net_component(self, **nets):
        """ initialise a network component """
        for name, net in nets.items():
            # info 
            logging.debug(f'Initialzing {name} of {self.model_name}.')
            # init weights
            utils.initialize_weights_kaiming_normal(net)
            # cuda() or cpu()
            net.to(self.device)
            # print net arch info
            logging.debug( utils.network_num_parameters(net) )
            logging.debug( str(net) )


    @abstractmethod
    def init_optimizer(self):
        raise NotImplementedError
#         beta1, beta2 = 0.5, 0.99
#         self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = 0)
#         self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(beta1, beta2), weight_decay = 0)
        
#         utils.print_line()
#         print('Use ADAM optimizers for G and D.')
        
        
    def set_fixed_noise(self, num_visual_samples):
        """ reset the numbe of fixed visual samples """
        self.num_visual_samples = num_visual_samples
        self.fixed_noise = self.get_noise( num_visual_samples )
    
    
# ================ training part ================

    def get_noise(self, num_ = 1):
        """ get the noise in the hidden space of the predefined distribution """
        out_noise = torch.randn(num_, self.hidden_dim, device=self.device)
        return out_noise
    
    
    @abstractmethod
    def stepTraining(self, batch_x):
        """ training of each step, implemented by the model """
        raise NotImplementedError


# ================ sampling part ================

    def sample(self, in_noise):
        """ sample num_visual_samples images from current G model. """
        assert in_noise.size(1) == self.hidden_dim
        self.G.eval()
        in_noise = in_noise.to(self.device)
        with torch.no_grad():
            samples = self.G( in_noise )
        return samples.cpu()

    def sampleN(self, N):
        """ N random samples from self.sample(). """
        return self.sample( self.get_noise( max(N, 1) ) )

    sampleOne = partialmethod(sampleN, N = 1)

    def sample_fixed(self):
        """ sample from the fixed noise """
        return self.sample(self.fixed_noise)    

    def sample_yielder(self, num = 0):
        """ a python generator of the generator. Gives a random new sample. """
        if num == 0:
            while True:
                yield self.sampleOne()
        else:
            for _ in range(num):
                yield self.sampleOne()
    
    
    def encode(self, X):
        """ encode and decode the samples X 
        with the encoder and the decoder of the model.
        """
        if not hasattr(self, 'E'):
            logging.warning(f'{self.__name__} does not have Encoder.')
            return None
        self.E.eval()
        X = X.to(self.device)
        with torch.no_grad():
            z = self.E( X )
            mu, log_var = torch.chunk(z, 2, dim=1)  # mean and log variance.
        return mu.cpu()
    
    
    def reconstruct(self, X):
        """ encode and decode the samples X 
        with the encoder and the decoder of the model.
        """
        if not hasattr(self, 'E'):
            logging.warning(f'{self.model.__name__} does not have Encoder.')
            return None
        self.E.eval()
        self.G.eval()
        X = X.to(self.device)
        with torch.no_grad():
            z = self.E( X )
            mu, log_var = torch.chunk(z, 2, dim=1)  # mean and log variance.
            samples = self.G( mu )
        return samples.cpu()
    
    
    def latent_traversal_dim(self, dim, num_range = 61):
        """
        Generate the samples when the the specified hidden code varies.
        Return a list of torch tensors. 
        num_range : the num of code to change
        """

        code_variant = torch.linspace(-self.traversal_code_limit, self.traversal_code_limit, num_range).to(self.device)
        zeros = torch.zeros(1, self.hidden_dim, device = self.device)
            
        images = []
        # each sub picture
        for varying_code in code_variant:
            this_code = torch.cat( [ self.fixed_noise.clone(), zeros ], 0 )
            this_code[:, dim] = varying_code

            samples = self.sample(this_code)
            images.append(samples)

        return images
    
    
    def latent_traversal_given_samples_dim(self, X, dim, num_range =61):
        """
        Reconstruct the sample sequence when the the specified hidden code varies.
        Return a list of torch tensors. 
        num_range : the num of code to change
        """
        
        # encode the samples
        codes = self.encode(X)
        
        print(self.model_name, self.traversal_code_limit)
        code_variant = torch.linspace(-self.traversal_code_limit, self.traversal_code_limit, num_range).to(self.device)
        # reconstruct
        images = []
        # each sub picture
        for varying_code in code_variant:
            this_code = codes.clone().to(self.device)
            this_code[:, dim] = varying_code

            samples = self.sample(this_code)
            images.append(samples)

        return images

        

# ================ Save / Load part ================

    def save(self, save_path):
        """ save models in the specific save path """
        for name, model in self.name_model_dict.items():
            torch.save(model.state_dict(), os.path.join(save_path, f"{name}.pkl") )

        logging.info("Models saving completed!")


    def load(self, save_path):
        """ load models in the specific save path """
        flag = True
        for name, model in self.name_model_dict.items():
            try:
                model.load_state_dict( torch.load(os.path.join(save_path, f"{name}.pkl")) )
            except FileNotFoundError as e:
                logging.critical(f'The model {name} is not found!')
                flag = False
        
        if flag:
            logging.info("Models loading completed!")

        