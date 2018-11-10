""" 
Python 3.6 
PyTorch 0.4
"""

import os
import logging
import math
from configparser import ConfigParser
import functools, itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from tqdm import tqdm_notebook

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX

  
def print_line(num = 1):
    for _ in range(num):
        logging.debug('================================')


def set_logging(level = logging.INFO):
    """
    set logging level and format.
    """
    logging.basicConfig(level = level, filename = '', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def read_config(dataset):
    """
    Read from the config file to get the parameters.
    """
    from Models.AbstractModel import Models
    dataset = dataset.lower()
    assert dataset in Models.keys()
    config = ConfigParser()
    config.read('configuration.ini')
    
    toreturn = {'dataset':dataset}
    for key, val in dict(config[dataset]).items():
        try:
            toreturn[key] = int(val)
        except:
            toreturn[key] = val

    logging.critical(f'I will train for {toreturn["epoch"]} epochs!')
    return toreturn


def long_tensor_to_onehot(idx, max_idx):
    """ from a one-dimension LongTensor to get the onehot vector. 
    >>> long_tensor_to_onehot(torch.LongTensor([1,0,2]), 3)
    tensor([[ 0,  1,  0],
            [ 1,  0,  0],
            [ 0,  0,  1]])
    """
    return torch.zeros(idx.size()[0], max_idx).scatter_(1, idx.view(-1,1), 1).long()


def gen_random_labels(num_instance, max_idx):
    """ get random labels in randomrange(max_idx)
    >>> labels = gen_random_labels(10, 3)
    """
    return torch.multinomial(torch.ones(max_idx), num_instance, replacement = True)
    

def network_num_parameters(net):
    """
    Compute the number of parameters
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logging.debug(f'Total number of parameters: {num_params}')
    return num_params

    
def to_np(x):
    """ transform a given tensor to numpy array
    >>> to_np(torch.ones(3,2))
    array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)
    """
    if not torch.is_tensor(x):
        raise TypeError('We need tensor here.')
        
    return x.to( torch.device('cpu') ).numpy() 

 
    
def cov(x):
    """ calculate covariance matrix of rows 
    >>> a = torch.arange(18).view(3,6)
    >>> cov(a)
    tensor([[ 3.5000,  3.5000,  3.5000],
            [ 3.5000,  3.5000,  3.5000],
            [ 3.5000,  3.5000,  3.5000]])
    >>> t = torch.FloatTensor([[1,2,6],[2,4,0],[5,2,3],[9,5,1]])
    >>> cov(t)
    tensor([[  7.0000,  -4.0000,  -1.5000, -10.0000],
            [ -4.0000,   4.0000,  -1.0000,   4.0000],
            [ -1.5000,  -1.0000,   2.3333,   4.0000],
            [-10.0000,   4.0000,   4.0000,  16.0000]])

    """
    mean_x = torch.mean(x, 1, keepdim = True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    return c


def addGuassianNoise(x, sigma = 0.1):
    """ Add small Gaussian noise to input x 
    >>> tmp = addGuassianNoise(torch.ones(12,13), sigma = 1)
    """
    if not torch.is_tensor(x):
        raise TypeError('We need tensor here.')

    to_add = torch.randn(x.size()) * sigma

    return x + to_add


def check_nan(dict_):
    """
    Check nan for a dict.
    >>> dict_ = {'a': 0, 'b': 1, 'c': 0, 'd': 10}
    >>> check_nan(dict_)
    True
    >>> from math import nan
    >>> check_nan({'a': 0, 'b': nan, 'c': 0, 'd': nan}) 
    ArithmeticError: We find the loss(es) [b, d] nan in PairwiseDis.
    """
    # check nan
    flag_nan = [ key for key, loss in dict_.items() if math.isnan(loss) ]
    if len(flag_nan) > 0:
        str_keys_nan = ', '.join(flag_nan)
        raise ArithmeticError(f'We find the loss(es) [{str_keys_nan}] nan.')
    return True


def pair_iters(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
    
    
# ================================ Visualization part ================================

def show_image(x, nrow = 0):
    """ Show the images (in tensors) in a grid where number of row specified nrow
    >>> import torch; show_image(torch.rand(25, 3, 32, 32), nrow = 5)
    """
    if nrow == 0:
        nrow = max( int(np.sqrt(len(x))), 6 )

    x.detach_()
    x = x.cpu()

    x = torchvision.utils.make_grid(x, nrow=nrow, normalize=True, range=(-1,1), padding=4 ).float()
#     x = torchvision.utils.make_grid(x, nrow = nrow, padding = 4, scale_each=True, normalize = True) 
    x = to_np(x)
    
    if len(x.shape) == 4:
        x = x.transpose(0,2,3,1)
    elif len(x.shape) == 3:
        x = x.transpose(1,2,0)
    
    f, ax = plt.subplots(figsize=(12, 9), dpi=300)
    plt.imshow(x)
    plt.show()


def generate_animation(imgs, save_path, name=None):
    """
    Generate a gif animation of given imgs.
    >>> import torch; generate_animation([torch.randn(3, 32, 32) for _ in range(10) ], 'save_dir/unittest')
    """
    if torch.is_tensor(imgs):
        imgs.requires_grad_(False)
    
    normalize = functools.partial( torchvision.utils.make_grid, nrow=6, normalize=True, range=(-1,1), padding=4 )
    images = [ to_np(normalize(img)*255).astype(np.uint8).transpose(1, 2, 0) for img in imgs  ]
    
    if name is None:
        name = 'generated_animation'
    imageio.mimsave( os.path.join(save_path, f'{name}.gif'), images, fps = 40)
    


def exp_encoded_range_visual(sample, models, code_dim_varying, inverse = [False]*5, num_ = 6, filename = 'imgs/tmp.png'):
    """
    Generate a sequence of samples by varying the encoded code given a sample
    >>> from Dataset import random_samples
    >>> sample = random_samples('mnist')
    >>> from Models.Beta_VAE import Beta_VAE; models = [Beta_VAE(1 , dataset = 'mnist', hidden_dim = 10, gpu_mode = False)]
    >>> code_dim_varying = [0,5,7]
    >>> out = exp_encoded_range_visual(sample, models, code_dim_varying, [False], 3, 'save_dir/unittest/exp_encoded_range_visual.png')
    >>> out.type()
    'torch.FloatTensor'
    """
    
    assert len(sample) == 1
    samples_batch = []

    with torch.no_grad():
        
        for i0, (model, code) in tqdm_notebook(enumerate(zip(models, code_dim_varying)), 'exp_encoded_range_visual'):
            if 'infogan' in model.model_name.lower():
                continue
#             print(model.model_name, code)
            
            samples = model.latent_traversal_given_samples_dim(sample, code, num_range=6)
            if inverse[i0]:
                samples = samples[::-1]
            samples_this_batch = torch.cat(samples, 0)
            samples_batch.append(samples_this_batch)
    
    samples = torch.cat(samples_batch, 0)
    show_image(samples, nrow = num_ )
    
    tmp = [ torchvision.utils.make_grid(s, nrow = num_, padding = 4, normalize=True, range=(-1,1), pad_value=1 ).unsqueeze(0) for s in samples_batch ]
    samples = torch.cat(tmp, 0)
    torchvision.utils.save_image(samples, filename = filename, nrow = 1, padding=2, pad_value=1)
    logging.info('Files saved! ' + filename)
    
    return samples
    
    
# ================================ Recording part ================================


def updateDataFrame(index, in_dict, label):
    '''
    Save the dict info in the csv file specified by label, return the DataFrame
    >>> updateDataFrame('label', {'s':2, 't':8}, 'doctest')
           s  t
    label  2  8
    '''
    
    # save path
    save_path = 'PD_DF'
    csv_file = os.path.join(save_path, f'{label}.csv')
    # load the existing dataframe
    df = loadDataFrame(label)
    # make the dataframe
    new_entry = pd.DataFrame.from_dict({index:in_dict}, orient = 'index')

    # if not dataframe exists
    if df is None:
        df = new_entry
        # if an old entry is in the dataframe
    else:
        if index in df.index:            
            df.loc[index] = new_entry.loc[index]
        else:
            # this is a new entry
            df = df.append(new_entry, verify_integrity=True, sort=False)

    # save it
    df.to_csv(csv_file, header='dataframe')
    
    return df


def loadDataFrame(label):
    '''
    Load the dataframe from csv specified by the label
    >>> loadDataFrame('doctest')
           s  t
    label  2  8
    >>> loadDataFrame('some_label_no_exist')
    Target filed not found.
    '''
    # save path
    save_path = 'PD_DF'
    csv_file = os.path.join(save_path, f'{label}.csv')
    # load it
    try:
        df = pd.read_csv(csv_file, index_col=0)
        return df
    except Exception as e:
        print('Target filed not found.')
        return None


# ================================ Dist and distributions part ================================


def pdist(A, B, sqrt=True):
    """ Pairwise Euclidean distance
    >>> import torch; A = torch.randn(2,4); B = torch.randn(15,4); pdist(A, B).size()
    torch.Size([2, 15])
    """
    A_squared = A.pow(2).sum(1).unsqueeze(1)
    B_squared = B.pow(2).sum(1).unsqueeze(0)
    AB = torch.mm(A, B.t()) # A @ B.t()
    A_B_squared = A_squared + B_squared - 2 * AB
    A_B_squared.clamp_(min=1e-16) # min=0 will cause Nan grad
    return torch.sqrt( A_B_squared ) if sqrt else A_B_squared


def log_prob_standard_normal(x):
    """ The log probability of x in the standard normal distribution.
        asser x.dim in {1,2}
        >>> import torch;from torch.distributions.multivariate_normal import MultivariateNormal
        >>> normal_prior = MultivariateNormal(torch.zeros(5), torch.eye(5) )
        >>> x = torch.randn(15, 5) * 2 + 4
        >>> (log_prob_standard_normal(x) - normal_prior.log_prob(x)).norm(p=1).item() < 1e-8
        True
        
        # Another implementation
        device = x.device
        dim = x.size()[1]
        normal_prior = MultivariateNormal(torch.zeros(dim, device=device), \
                                          torch.eye(dim, device=device) )
        # probability of p(x)
        log_prob_p = normal_prior.log_prob(x) # B
        
    """
    if x.dim() == 1: x.unsqueeze_(1)
    dim = x.size()[1]
    d_log_2pi = dim * math.log(2 * math.pi)
    return - 0.5 * ( x.pow(2).sum(1) + d_log_2pi )
    

def log_prob_of_multiple_dist(mu, log_var, x):
    """ Compute the log probabilty of x subject to distributions define by mu and log_var
    Args:
    mu     - N * d
    logvar - N * d
    x      - B * d

    -0.5 * \|x-mu\|^2 / log_var.exp() - 0.5 * log_var.sum() - 0.5 * log(2*pi) * d
    
    return B * N

    >>> import torch; from torch.distributions import Normal
    >>> n = Normal(torch.zeros(1), torch.ones(1))
    >>> x = torch.randn(1) * 5 + 4
    >>> pred = log_prob_of_multiple_dist(torch.zeros(1), torch.zeros(1), x)
    >>> (pred - n.log_prob(x) ).abs().item() < 1e-8
    True
    """

    r"""
    This is the simple implementation. Out implementation is 20x faster on CPU and 1000x faster on GPU.

    def t(mu, log_var, x):
        tmp = []
        for mu_, log_var_ in zip(mu, log_var):
            mn = MultivariateNormal(mu_, torch.diag( log_var_.exp() ))
            ground_truth = mn.log_prob(x) # B 
            tmp.append(ground_truth)

        # B * N
        out = torch.stack(tmp, dim = 1) 
    """
    
    if mu.dim()==1: mu.unsqueeze_(1)
    if log_var.dim()==1: log_var.unsqueeze_(1)
    if x.dim()==1: x.unsqueeze_(1)
    
    dim = mu.size()[1]

#         # N * d, mu^2
#         mu.pow(2) 
#         # N * d, mu^2 ./ var
#         mu.pow(2).div( log_var.exp() )
#         # N * 1, \sum_i mu_i^2/var_i
#         mu.pow(2).div( log_var.exp() ).sum(1).unsqueeze(1)

#         # B * d, x^2
#         x.pow(2) 
#         # 1 * B * d, x^2
#         x.pow(2).unsqueeze(0)
#         # N * 1 * d, var
#         var = log_var.exp().unsqueeze(1)
#         # N * B * d, x^2 ./ var
#         x.pow(2).unsqueeze(0).div( var )
#         # N * B, \sum_i x_i^2/var_i
#         x.pow(2).unsqueeze(0).div( var ).sum(2)

#         # N * B, \sum_i mu_i*x_i/var_i
#         torch.mm(mu.div(log_var.exp), B.t())

    # N * d
    var = log_var.exp() + 1e-8
    
    mu_div_var_squared = mu.pow(2).div( var ).sum(1).unsqueeze(1) # N * 1
    x_div_var_squared  = x.pow(2).unsqueeze(0).div( var.unsqueeze(1) ).sum(2) # N * B
    mu_x_div_var_squared = torch.mm(mu.div( var ), x.t()) # N * B

    # N * B
    mu_x_var_dist = mu_div_var_squared + x_div_var_squared - 2 * mu_x_div_var_squared
    mu_x_var_dist.clamp_(min=10**-16)

    # N * 1
    log_var_sum = log_var.sum(1, keepdim = True)

    # 0-dim
    d_log_2pi = dim * math.log(2 * math.pi)
    
    # N * B
    out = -0.5 * (d_log_2pi + log_var_sum + mu_x_var_dist)

    return out.t() # B * N

 
# ================================ Initialization part ================================    
    

def initialize_weights_kaiming_normal(*nets):
    for net in nets:
        for m in net.modules():
            if isinstance( m, (nn.Conv2d,nn.Linear,nn.ConvTranspose2d) ):
                if hasattr(m,'weight'):
                    torch.nn.init.kaiming_normal_(m.weight.data)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
    logging.debug('Weight initialized with kaiming_normal')
    

# ================================ Breakpoint tool part ================================
    
def breakpoint():
    ''' to place break point somewhere '''
    from IPython.core.debugger import set_trace;set_trace()
