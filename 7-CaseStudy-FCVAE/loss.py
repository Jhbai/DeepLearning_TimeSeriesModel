import torch
import numpy as np

def log_likelihood(mean, variance, target):
    """
    :param mean: The prediction mean value
    :param variance: The prediction variance
    :param target: The ground truth
    :return: The log likelihood value
    """
    LL = torch.sum(-.5*torch.log(2*np.pi*variance) - (target - mean)**2/(2*variance), dim = 1)
    return torch.mean(LL)

def kl_divergence(z_mean, z_logvar):
    """
    Compute the KL-Divergence of latent space w.r.t N(0, 1)
    
    
    :param z_mean: Mean of Latent, (n_batch, n_zdim)
    :param z_logvar: Log variance of Latent, (n_batch, n_zdim)
    :return: A single value of KL-Divergence in torch.tensor
    """
    kl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
    return kl.mean()  # 對 batch 求平均

def criterion(mean, logvar, z_mean, z_logvar, x, beta=.5):
    """
    Compute the reconstruct error via negative log_likelihood and KL-Divergence for VAE
    
    :param mean: The decoder for mean of the reconstruction
    :param logvar: The decoder for log_variance of the reconstruction
    :param z_mean: The mean of latent
    :param z_logvar: The log_variance of latent
    :param x: Raw data
    :param beta: The weight of KL_Divergence
    :return: The single value of total loss of the model
    """
    # Recons-Loss
    recon_loss = -log_likelihood(mean, logvar.exp(), x).mean()
    
    # KLDiv-Loss
    kl_div = kl_divergence(z_mean, z_logvar)
    
    # Total_Loss
    loss = recon_loss + beta * kl_div
    return loss
