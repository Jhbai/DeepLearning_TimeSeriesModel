import torch

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, sigma=None):
    """
    計算高斯核矩陣
    :param x: 張量 (batch_size, features)
    :param y: 張量 (batch_size, features)
    :param kernel_mul: 核帶寬調節參數
    :param kernel_num: 使用的高斯核數量
    :param sigma: 核的標準差，若為 None 則自動計算
    :return: 核矩陣 (batch_size, batch_size)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    total = torch.cat([x, y], dim=0)  # 合併 x 和 y
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if sigma is None:
        sigma = torch.mean(L2_distance).detach()
    sigma = sigma.item()

    bandwidth = sigma * kernel_mul ** torch.arange(0, kernel_num).float().to(x.device)
    kernel_val = [torch.exp(-L2_distance / bandwidth[i]) for i in range(kernel_num)]
    return sum(kernel_val)

def MMD(x, y, kernel_mul=2.0, kernel_num=5, sigma=None):
    """
    計算 MMD
    :param x: 張量 (batch_size, features)
    :param y: 張量 (batch_size, features)
    :return: MMD 損失值
    """
    kernels = gaussian_kernel(x, y, kernel_mul, kernel_num, sigma)
    xx = kernels[:x.size(0), :x.size(0)]
    yy = kernels[x.size(0):, x.size(0):]
    xy = kernels[:x.size(0), x.size(0):]
    return torch.mean(xx + yy - 2 * xy)
