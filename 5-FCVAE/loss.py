import torch
def kl_divergence(z_mean, z_logvar):
    """
    計算 KL 散度
    z_mean: 潛變量的均值 (batch_size, z_dim)
    z_logvar: 潛變量的對數方差 (batch_size, z_dim)
    """
    kl = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1)
    return kl.mean()  # 對 batch 求平均

def vae_loss(x, mean, logvar, z_mean, z_logvar, beta=1.0):
    """
    VAE 的損失函數
    x: 原始數據
    mean, logvar: 解碼器的均值和對數方差
    z_mean, z_logvar: 編碼器生成的潛變量均值和對數方差
    beta: KL 散度的權重
    """
    # 重建損失
    recon_loss = -log_likelihood(x, mean, logvar).mean()
    
    # KL 散度
    kl_div = kl_divergence(z_mean, z_logvar)
    
    # 損失函數
    loss = recon_loss + beta * kl_div
    return loss
