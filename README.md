# Pairwise-Indepence-Autoencoder: learning disentangled representation with pairwise independence

## [ [Paper]](https://arxiv.org/abs/1811.) 
The implementation of PIAE. 
In this work we introduce the method to learn disentangled representation unsupervisedly with the assumption that the latent codes are pairwise independent. We also give our implementation of other learning methods..

## Examples of latent traversal of PIAE(MI)

### PIAE(MI)

![PIAEMI_brightness](/imgs/code003.gif)
Brightness

![PIAEMI_gender](/imgs/code040.gif)
Gender

### PIAE(KL)

![PIAEKL_brightness](/imgs/code039.gif)
Brightness

![PIAEKL_gender](/imgs/code013.gif)
Gender

## Prerequisites
- python 3.6
- pytorch 0.4.1, torchvision 0.2.1, tensorboardX
- numpy, scipy, scikitlearn, matplotlib, pandas, tqdm, imageio, jupyter


## Run
To train and test the model, please follow the guide in main.ipynb.
