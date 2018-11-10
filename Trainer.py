""" 
Python 3.6 
PyTorch 0.4
"""

import os, gc
import time, datetime
import logging

from tqdm import tnrange, tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import imageio

from sklearn.cross_decomposition import CCA
from sklearn.model_selection import cross_val_score 
import torch
import torchvision
import tensorboardX

import Dataset, utils
import SubspaceScore



class Trainer(object):
    ''' The Trainer for generative models. '''
    
# ================================ Initialization part ================================

    def __init__(self, model, dataset, epoch, batch_size = 64, **kwargs):
        """ 
        The parameter model is an instance of the abstract class AbstractModel,
        which is the model used to fit the data and sample from the distribution.
        """
        
        # configuration
        self.save_dir = 'save_dir'
        self.dataset = dataset
        
        # training parameter
        self.num_epoch, self.batch_size = epoch, batch_size
        
        # model
        self.model = model
        
        # device
        self.device = self.model.device

        # data
        self.dataset_transform = 'tanh'
        assert self.model.tanh
        self.trainloader = Dataset.load_dataset(self.dataset, 
                                                batch_size = self.batch_size, 
                                                transform = self.dataset_transform)

        # dict to record loss and other info
        self.record_dict = { }
        
        # other init
        self.num_visual_samples = 36
        self.model.set_fixed_noise(self.num_visual_samples)
        
        # saving path
        self.defineSavePath()
        
        # tensorboardX
        self.tbx_writer = tensorboardX.SummaryWriter()
#         logging.info(f'tensorboard --logdir runs')
        
        # summary
        summary = f'''
              model: {self.model.model_name}, 
               tanh: {self.model.tanh},
           net_arch: {self.model.net_arch},
            dataset: {self.dataset}, 
          save_path: {self.save_path}, 
         hidden_dim: {self.model.hidden_dim}
                   '''
        self.tbx_writer.add_text('summary', summary )
        logging.info("I summarize the configuration as follows.")
        logging.info(summary)
        
        
        logging.debug('Trainer init completed!')
        

    def defineSavePath(self):
        """ Define the save path for this experiment """
        self.save_path = os.path.join(self.save_dir, self.dataset, self.model.model_name)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        return self.save_path
    
    
    def train(self,  **kwargs): # test covered
        ''' 
        The main training process of the given model.
        '''
        
        save_interval = 1
        visualize_interval = 1
        
        utils.print_line()
        logging.debug('Training begins.')
        
        total_step = len(self.trainloader)
        
        # epoch loop
        for epoch in tnrange(self.num_epoch, desc='epoch loop'):
        
            # for each training step
            for step, (batch_x, _) in tqdm_notebook(enumerate(self.trainloader, 0), desc=f'{total_step} step loop', leave=False):  
                
                # preprocess
                batch_x = batch_x.float()
                
                # step train
                loss_dict = self.model.stepTraining(batch_x)
                
                # chk nan
                utils.check_nan(loss_dict)
                
                # write loss in tbx
                if ((step + 1) % 50) == 1:
                    epoch_step = epoch * total_step + step
                    for key,val in loss_dict.items():
                        self.tbx_writer.add_scalar(key, val, epoch_step)

                    # and in record dict
                    self.record_values(loss_dict)
                
                # print loss values
                if ((step + 1) % 100) == 1:
                    logging.debug(f"Epoch: [{epoch+1}] Step: [{step+1}/{total_step}]")
                    logging.debug(str(loss_dict))

            # end step loop

            # print epoch and get images in the TBX writer
            self.tbx_writer.add_text('Epoch', str(epoch), epoch)
            self.tbx_writer.add_image('Generated_images_random', self._make_grid(self.model.sampleN( self.num_visual_samples )), epoch)
            self.tbx_writer.add_image('Generated_images_fixed' , self._make_grid(self.model.sample_fixed()), epoch )
            self.tbx_writer.add_image('Real_images_for_comparison' , self._make_grid(batch_x), epoch )
            # if this is the AE model, show the reconstructed images
            if hasattr(self.model, 'E'):
                self.tbx_writer.add_image('Reconstructed_images' , self._make_grid(self.model.reconstruct(batch_x)), epoch )
                self.tbx_writer.add_image('images_residual' , (self._make_grid(batch_x) - self._make_grid(self.model.reconstruct(batch_x))), epoch )
                

            # save the net model
            if save_interval > 0 and (epoch+1) % save_interval == 0: 
                self.save()
            
            # visualize with matplotlib
            if visualize_interval > 0 and (epoch+1) % visualize_interval == 0: 
                self.visualize_results(epoch+1)
            
        # end epoch loop
        
        # savings
        self.save()
        
        logging.info("Models saved. Training finished!")
        self.tbx_writer.add_text('Epoch', "Training finished!", self.num_epoch+1) 
        
        # gc
        torch.cuda.empty_cache()
        gc.collect()

        return self
    
    # end train()


# ================================ Save / Load part (test covered) ================================
        
    def save(self):
        save_path = self.save_path
        self.model.save(save_path)

        logging.debug("Models saving completed!")
        
        return self
        
        
    def load(self):
        logging.info("I try to load the existing model.")
        self.model.load(self.save_path)
        logging.debug('Models loading completed!')
        
        return self
    
    
    def record_values(self, kwargs):
        ''' record the values. '''
            
        for key in kwargs.keys():
            self.record_dict.setdefault(key, [])
            self.record_dict[key].append(kwargs[key])
            
        return self


    
# ================================ Visualization part (test covered) ================================
    
    def visualize_results(self, epoch, N = 0):
        """ visualize the sampled results and save them"""

        save_path = os.path.join(self.save_path, 'visualization')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        samples = self.model.sample_fixed() if N == 0 else self.model.sampleN(N)

        samples_grid = self._make_grid(samples)

        plt.imshow( utils.to_np(samples_grid).transpose(1, 2, 0) )
        plt.show()

        torchvision.utils.save_image(samples_grid, filename = os.path.join(save_path, 'epoch%03d' % epoch + '.png') )

        logging.info('Sampled images saved.')
        
        return self


    def _make_grid(self, samples):
        """ Easily make grids of images from samples. """
        return torchvision.utils.make_grid(samples, nrow=max( int(np.sqrt(len(samples))), 6 ), normalize=True, range=(-1,1), padding=4 ).float()
    
    
    def latent_traversal(self, samples = None, epoch = 0, save_path = None):
        if save_path is None:
            save_path = os.path.join(self.save_path, 'visualization')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i0 in tnrange(self.model.hidden_dim, desc = 'latent traversal', leave = False):
            if samples is None:
                images = self.model.latent_traversal_dim(i0) # list T, tensor B C H W
            else:
                images = self.model.latent_traversal_given_samples_dim(samples, i0) # list T, tensor B C H W

            epoch_path = os.path.join(save_path, f'latent_traversal_{epoch}')
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            utils.generate_animation(images, epoch_path, f'code{i0:03}')
        
        return self

    
# ================================ Evaluation part ================================

    
    def subspace_score(self, epoch = 0): # test covered
        """ get the subspace score of the generative model
        """
        mean_score, score_batch, coefficient_matrix_sym_batch, reconstructed_accuracy_batch, NMI_batch = SubspaceScore.subspace_score(self.model, self.trainloader)
        
        self.tbx_writer.add_text('Mean subspace score', f'{mean_score}.', epoch)
        self.tbx_writer.add_text('subspace scores', f'{ str(score_batch) }.', epoch)
        self.tbx_writer.add_text('reconstructed_accuracy', f'{ str(reconstructed_accuracy_batch) }.', epoch)
        self.tbx_writer.add_text('NMI', f'{ str(NMI_batch) }.', epoch)
        
        
        for i0, coefficient_matrix_sym in enumerate(coefficient_matrix_sym_batch, 0):
            self.tbx_writer.add_image(f'coefficient_matrix{epoch}', torch.from_numpy(coefficient_matrix_sym).unsqueeze(0), i0)
        
        return mean_score, score_batch, reconstructed_accuracy_batch, NMI_batch

