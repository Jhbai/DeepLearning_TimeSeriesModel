import torch
import torch.nn as nn

def softmin(x, gamma):
    """
    Soft-min 操作，平滑最小值
    :param x: 張量
    :return: Soft-min 的結果
    """
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)

def SoftDTW(X, Y, gamma = 1.):
    """
    初始化 Soft-DTW 模組和計算 Soft-DTW
    :param X: 序列 X，形狀為 (batch_size, seq_len, features)
    :param Y: 序列 Y，形狀為 (batch_size, seq_len, features)
    :return: Soft-DTW 距離
    """
    batch_size, seq_len_x, _ = X.size()
    _, seq_len_y, _ = Y.size()
    D = torch.cdist(X, Y, p=2)
    # 初始化累積成本矩陣
    R = torch.zeros(batch_size, seq_len_x + 1, seq_len_y + 1).to(X.device)
    R[:, 0, :] = float('inf')  # 第一列設為無窮大
    R[:, :, 0] = float('inf')  # 第一行設為無窮大
    R[:, 0, 0] = 0  # (0, 0) 初始化為 0
    
    # 動態規劃計算 Soft-DTW
    for i in range(1, seq_len_x + 1):
        for j in range(1, seq_len_y + 1):
            r = torch.stack([
                R[:, i - 1, j],    # 從上方到達
                R[:, i, j - 1],    # 從左側到達
                R[:, i - 1, j - 1] # 從左上角到達
            ], dim=-1)
            R[:, i, j] = D[:, i - 1, j - 1] + softmin(r, gamma)
    
    # 返回最後的 Soft-DTW 距離
    return torch.mean(R[:, -1, -1])