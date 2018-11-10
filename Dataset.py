""" Module to prepare dataset loaders. """

import os
import logging

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

import utils


def transform_mapping(trans, istensor=False, resize=None):
    """ get the transform of the given type 
    >>> transform_mapping('01', False)
    Compose(
        ToTensor()
    )
    >>> transform_mapping('01', True)
    Compose(
        ToTensor()
    )
    >>> transform_mapping('01', True, 61)
    Compose(
        Resize(size=(61, 61), interpolation=PIL.Image.BILINEAR)
        ToTensor()
    )
    >>> transform_mapping('tanh', False)
    Compose(
        ToTensor()
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    )
    >>> transform_mapping('tanh', False, 67)
    Compose(
        Resize(size=(67, 67), interpolation=PIL.Image.BILINEAR)
        ToTensor()
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    )
    >>> transform_mapping('pretrained_model', True)
    Compose(
        ToPILImage()
        Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR)
        ToTensor()
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    >>> transform_mapping('inception', False)
    Compose(
        Resize(size=(299, 299), interpolation=PIL.Image.BILINEAR)
        ToTensor()
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    """

    available_transforms = ['tanh', '01', 'pretrained_model', 'inception']
    trans = trans.lower()
    assert trans in available_transforms
    
    # diff transforms
    transform_dict = {
        '01':   [ transforms.ToTensor(), ],
        'tanh': [ transforms.ToTensor(), transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3 ) ],
        'pretrained_model': [ transforms.Resize( (224, 224) ), transforms.ToTensor(), \
                              transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )  ],  
        'inception':        [ transforms.Resize( (299, 299) ), transforms.ToTensor(), \
                              transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )  ],  
    }
    
    trans_list = transform_dict[trans]
    
    # add Resize or ToPILImage
    if trans in ('tanh', '01') and resize!=None:
        trans_list = [ transforms.Resize( (resize, resize) ) ] + trans_list
    elif trans in ('pretrained_model', 'inception') and istensor:
        trans_list = [ transforms.ToPILImage() ] + trans_list
    
    # merge together
    toreturn = transforms.Compose(trans_list)

    return toreturn


def load_dataset(dataset, batch_size = 64, transform = 'tanh'):
    """
    Load dataset loader
    >>> trainloader = load_dataset("MNIST", 24)
    >>> trainloader.batch_size
    24
    >>> l = load_dataset('mnist', 5, 'tanh'); l.dataset
    Dataset MNIST
        Number of datapoints: 60000
        Split: train
        Root Location: ./data/mnist
        Transforms (if any): Compose(
                                 ToTensor()
                                 Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                             )
        Target Transforms (if any): None
    """
    dataset = dataset.lower()
    dataset_mapping = {'small-mnist': load_dataset_MNIST,
                        'mnist': load_dataset_MNIST,
                       'fashion_mnist': load_dataset_FashionMNIST,
                       'cifar10': load_dataset_CIFAR10,
                      }

    assert dataset in dataset_mapping.keys()

    if dataset == 'small':
        trainloader, _ = load_dataset_Small(batch_size, for_tanh = (transform == 'tanh') )
    elif dataset == 'cifar10':
        trainloader, _ = load_dataset_CIFAR10(batch_size, transform = transform_mapping(transform, False, 64) )
    else:
        trainloader, _ = dataset_mapping[dataset](batch_size, transform = transform_mapping(transform, False) )
    
    utils.print_line()
    dict_transform_str = {'tanh':'in [-1, 1].', 
                          '01':'in [0,1].', 
                          'pretrained_model': 'for pretrained models.', 
                          'inception': 'for inception.'}
    logging.info(dataset.upper() + ' Data loaded in normalized range ' + dict_transform_str[transform])
    
    return trainloader


def random_samples(dataset, n = 1, transform = 'tanh'):
    """
    load dataset loader
    >>> sample = random_samples("MNIST")
    >>> sample.size()
    torch.Size([1, 1, 28, 28])
    """
    trainloader = load_dataset(dataset, n, transform)
    return next(iter(trainloader))[0]



def get_data_loader(batch_size, trainset, testset = None, num_train=0, num_test=0, num_workers=4):
    """ get dataset loaders """
    # only take a small part
    if num_train != 0 and num_test != 0:
        trainset.train_data = trainset.train_data[:num_train, :, :]
        trainset.train_labels = trainset.train_labels[:num_train]

        testset.test_data = testset.test_data[:num_test, :, :]
        testset.test_labels = testset.test_labels[:num_test]
        
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=True)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             drop_last=True)

    return trainloader, testloader



def load_dataset_MNIST_small(batch_size, transform):
    trainloader, testloader = load_dataset_MNIST(batch_size = batch_size, 
                                                 num_train = 8192, 
                                                 num_test = 1000, 
                                                 transform = transform)
    logging.info('MNIST small version loaded')
    return trainloader, testloader


def load_dataset_MNIST(batch_size = 64, download=True, num_train = 60000, num_test = 10000, transform = None):
    """
    load the MNIST dataset
    >>> trainloader, testloader = load_dataset_MNIST()
    >>> trainloader.batch_size
    64
    """
    data_dir = os.path.join('.', 'data','mnist')

    trainset = torchvision.datasets.MNIST(root=data_dir, 
                                          train=True,
                                          download=download,
                                          transform=transform)
    
    testset = torchvision.datasets.MNIST(root=data_dir, 
                                         train=False,
                                         download=download,
                                         transform=transform)
    
    trainloader, testloader = get_data_loader(batch_size, trainset, testset, num_train, num_test, num_workers = 8)
    
    return trainloader, testloader


def load_dataset_FashionMNIST(batch_size = 64, download=True, num_train = 60000, num_test = 10000, transform = None):
    """
    load the FashionMNIST dataset
    >>> trainloader, testloader = load_dataset_FashionMNIST(34)
    >>> trainloader.batch_size
    34
    """    
    data_dir = os.path.join('.', 'data','fashion_mnist')

    trainset = torchvision.datasets.FashionMNIST(root=data_dir, 
                                                 train=True,
                                                 download=download,
                                                 transform=transform)
    
    testset = torchvision.datasets.FashionMNIST(root=data_dir, 
                                                train=False,
                                                download=download,
                                                transform=transform)
    
    trainloader, testloader = get_data_loader(batch_size, trainset, testset, num_train, num_test, num_workers = 8)
        
    return trainloader, testloader




def load_dataset_CIFAR10(batch_size=64, download=True, transform=None, ignore_label=False, num_workers=4):
    """
    load the CIFAR10 dataset
    >>> trainloader, testloader = load_dataset_CIFAR10(45)
    Files already downloaded and verified
    Files already downloaded and verified
    >>> trainloader.batch_size
    45
    """
    data_dir = os.path.join('.', 'data','CIFAR10')
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                          download=download,
                                          transform=transform)
    
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                         download=download,
                                         transform=transform)
    
    if ignore_label:
        trainset = IgnoreLabelDataset(trainset)
        testset = IgnoreLabelDataset(testset)
        
    trainloader, testloader = get_data_loader(batch_size, trainset, testset, num_workers=num_workers)
    
    return trainloader, testloader


    
