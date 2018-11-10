import logging

from tqdm import tnrange
import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.spatial import minkowski_distance
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression

import utils


def subspace_score(model, data_loader, num_tries = 5):
    """ get the subspace score of the model with the given data loader.
    NOTICE, the fit_self now uses least squared plus topk.
    >>> from Dataset import load_dataset; l = load_dataset('mnist', 8, 'tanh')
    >>> from Models.Beta_VAE import Beta_VAE; b = Beta_VAE(dataset = 'mnist', hidden_dim = 4, gpu_mode = False)
    >>> s = subspace_score(b, l)
    <BLANKLINE>
    """
#     # tries
#     num_tries = 5
    assert num_tries > 0
    # code variantion range
    num_range = 5
    code_variant = torch.linspace(-2, 2, num_range)
    # the number of samples per batch and the result
    num_sample_per_batch = 10
    # # fitting methods
    real_sample_fitting_method = 'LinearRegression'
    subspace_fitting_method = 'OMP'
    # balance between subspace independence and fitting fidelity
    ALPHA = 0.5
    # num of nonzeros in self regress
    K = num_range * num_sample_per_batch
    
    
    class Reconstructor():
        def __init__(self, reconstruct_method=None):
            super().__init__()

            self.lr = LinearRegression(fit_intercept=False, n_jobs = -1)
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

            
        def gels(self, X, Y):
            assert X.dim()==2
            assert Y.dim()==2
            assert X.size(1) == Y.size(1) # same dim
            num_x = len(X)

            Xtt = X.t().to(self.device) # dim * num_x
            Ytt = Y.t().to(self.device) # dim * num_y            
            coef, _ = torch.gels(Ytt, Xtt)
            coef = coef[:num_x]
            # projection
            Ytt_hat = torch.mm(Xtt, coef)
            Y_hat = Ytt_hat.t() # num_y * dim
            coef = coef.t() # num_y * num_x

            return coef, Y_hat
        
        def lrfit(self, X, Y):

            Xt, Yt = np.transpose(X), np.transpose(Y)
            coef = self.lr.fit(Xt, Yt).coef_
            # projection
            Y_hat = coef @ X # the projection, num_y * dim
            
            return coef, Y_hat
            

        def fit(self,X,Y):
            """
            Args:
            X : numpy.ndarray num_x * dim
            Y : numpy.ndarray num_y * dim
            """
            X, Y = normalize(X, axis=1), normalize(Y, axis=1) # unit length
            
            num_x = len(X)
            dim = X.shape[1]
            if dim >= num_x:
                X = torch.from_numpy(X)
                Y = torch.from_numpy(Y)
                coef, Y_hat = self.gels(X, Y)
                coef, Y_hat = coef.cpu().numpy(), Y_hat.cpu().numpy()
            else:
                coef, Y_hat = self.lrfit(X, Y)
                
            dist_to_projection = np.mean(minkowski_distance(Y, Y_hat))
            assert dist_to_projection <= 1
            
            return dist_to_projection, coef, Y_hat


        def fit_self(self,X):
            logging.debug(f'Reconstructor fit_self() called. Begin to fit {X.shape}' )
            X = normalize(X, axis=1) # unit length
            num_sample = len(X)
            dim = X.shape[1]
            all_idx = np.arange(num_sample)
            # the coef matrix to return
            result_matrix = np.zeros([num_sample, num_sample])
            # the coef matrix in torch
            result_matrix_t = torch.zeros(num_sample, num_sample).to(self.device)
            # the X in torch
            X_t = torch.from_numpy(X).to(self.device)
    
            # for each column ticked out
            for i1 in tnrange(num_sample, desc='fit_self', leave=False):
                # idx ticked
                this_idx = np.delete(all_idx, i1).tolist()
                if dim >= num_sample:
                    coef, _ = self.gels(X_t[this_idx, :], X_t[i1, :].unsqueeze(0))
                    coef.squeeze_()
                else:
                    coef, _ = self.lrfit(X[this_idx, :], X[i1, :].reshape(1, dim) )
                    coef = torch.from_numpy(coef).squeeze().to(self.device)

                # get the largest ones
                val, idx = coef.abs().topk(K)
                # reshape
                tmp = torch.zeros_like(coef)
                tmp.scatter_(0, idx, val*coef[idx].sign())
                # assign
                result_matrix_t[i1, this_idx] = tmp

            # as np
            result_matrix = utils.to_np(result_matrix_t)

            return result_matrix

    final_result_batch = []
    coefficient_matrix_sym_batch = []
    reconstructed_accuracy_batch = []
    NMI_batch = []
    
    # tries
    for i0 in tnrange(num_tries, desc = 'total rounds'):
        # normalizer
        scaler = StandardScaler(with_mean=True, with_std=False)

        # reconstructor 
        r_fit_real = Reconstructor(real_sample_fitting_method)
        r_fit_generated = Reconstructor(subspace_fitting_method)

        # get some of the real samples
        dataloader_iter = iter( data_loader )
        num_batch_real_samples = min(200, len(dataloader_iter))
        part_of_real_samples = torch.cat( [ next(dataloader_iter)[0] for _ in range(num_batch_real_samples) ] )
        part_of_real_samples_np = part_of_real_samples.view(part_of_real_samples.size()[0], -1).numpy()
        part_of_real_samples_np = scaler.fit_transform(part_of_real_samples_np) 

        # get the generated samples
        total_samples_batch = []
        total_labels_batch = []
        # generate sequences for every hidden dim
        for i1 in tnrange(model.hidden_dim, desc='Generating'):
            in_noise = model.get_noise(num_sample_per_batch).requires_grad_(requires_grad=False) 
            # each hidden code varies
            for i2 in range(num_range):
                in_noise[:, i1] = code_variant[i2]
                samples = model.sample( in_noise )
                total_samples_batch.append( samples )
                total_labels_batch.append( torch.ones(num_sample_per_batch) * i1 )

        # all the generated sequence
        total_samples = torch.cat(total_samples_batch)
        total_labels = torch.cat(total_labels_batch)

        # numpy format
        total_samples_np = utils.to_np( total_samples.view(total_samples.size()[0], -1) ) 
        total_samples_np = scaler.transform(total_samples_np)
        total_labels_np = utils.to_np( total_labels )
        
        logging.info(f'Begin to fit real samples in round {i0}.')
        # the reconstruction accuracy of the real samples from the generated samples
        reconstructed_accuracy = r_fit_real.fit(total_samples_np, part_of_real_samples_np)[0]
        
        logging.info(f'Begin to fit generated samples in round {i0}.')
        # fit the total_samples_np with itself
        coefficient_matrix = r_fit_generated.fit_self(total_samples_np)

        # symmetrization
        coefficient_matrix_abs = np.abs(coefficient_matrix)
        coefficient_matrix_sym = coefficient_matrix_abs + np.transpose(coefficient_matrix_abs)
        coefficient_matrix_sym_batch.append(coefficient_matrix_sym)
        
        # show the ``covariance'' matrix
#         plt.imshow( coefficient_matrix_sym / np.max(coefficient_matrix_sym) )
#         plt.show()
        
        # subspace clustering 
        label_hat = spectral_clustering(coefficient_matrix_sym, n_clusters = model.hidden_dim)
        NMI = metrics.normalized_mutual_info_score(label_hat, total_labels_np)
        
        # subspace score
        final_result_this =  NMI * ALPHA + (1 - reconstructed_accuracy) * (1 - ALPHA)

        to_print = f'''
                    ROUND {i0}: 
                    distance to projection:{reconstructed_accuracy} 
                    NMI:{NMI} 
                    final result:{final_result_this}
                    '''
        logging.info( to_print )

        final_result_batch.append(final_result_this)
        reconstructed_accuracy_batch.append(reconstructed_accuracy)
        NMI_batch.append(NMI)


    to_print = f'final result {np.mean(final_result_batch)}+-{np.std(final_result_batch)}'
    logging.info( to_print)
#     logging.info( str(data_loader.dataset) )
#     logging.info( model.model_name )

    return np.mean(final_result_batch), final_result_batch, coefficient_matrix_sym_batch, reconstructed_accuracy_batch, NMI_batch

