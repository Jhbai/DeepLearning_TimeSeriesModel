import torch
def anomaly_score(x, recon_mean, recon_logvar, z_mean, z_logvar):
    """
    計算異常分數
    x: 原始數據 (batch_size, n_dim)
    recon_mean: 解碼器輸出的均值 (batch_size, n_dim)
    recon_logvar: 解碼器輸出的對數方差 (batch_size, n_dim)
    z_mean: 編碼器的潛變量均值 (batch_size, z_dim)
    z_logvar: 編碼器的潛變量對數方差 (batch_size, z_dim)
    """
    # 1. 計算對數似然 -log p(x|z, c)
    recon_var = torch.exp(recon_logvar)
    log_likelihood = -0.5 * ((x - recon_mean) ** 2 / recon_var + torch.log(2 * torch.pi * recon_var))
    log_likelihood = log_likelihood.sum(dim=-1)  # 對所有維度求和
    anomaly_score = -log_likelihood  # 負對數似然作為異常分數

    return anomaly_score
