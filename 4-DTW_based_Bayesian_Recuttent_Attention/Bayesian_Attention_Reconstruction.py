import torch
import torch.nn as nn
from DTW import SoftDTW

def Loss(n_batch, y_pred, y_true, mean, logvar, beta = 0.003):
    # Variational Inference
    KL = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    DTWLoss = SoftDTW(y_pred, y_true)
    return beta*KL + DTWLoss


criterion = Loss

class Bayesian_Attention_Reconstruction_Model(nn.Module):
    def __init__(self, n_seq, n_hid, n_dim):
        super(Bayesian_Attention_Reconstruction_Model, self).__init__()
        self.k_w = nn.Linear(n_dim, n_hid)
        self.q_w = nn.Linear(n_dim, n_hid)
        self.v_w = nn.Linear(n_dim, n_hid)
        self.bias = nn.Parameter(torch.randn(size = (n_seq, n_hid)))
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(n_hid)
        nn.init.xavier_uniform_(self.k_w.weight)
        nn.init.xavier_uniform_(self.q_w.weight)
        nn.init.xavier_uniform_(self.v_w.weight)

        self.out_layer_mu = nn.Linear(n_seq*n_hid, n_seq*n_dim)
        self.out_layer_logvar = nn.Linear(n_seq*n_hid, n_seq*n_dim)

        self.n_seq, self.n_hid, self.n_dim = n_seq, n_hid, n_dim


    def forward(self, x):
        """
        執行Attention，學習Nadaraya Estimation的參數設定
        :param x: (n_batch, n_seq, n_dim)
        :return: (n_batch, n_seq, h_hid)
        """
        # Nadaraya Estimation
        k = self.k_w(x)
        q = self.q_w(x)
        v = self.v_w(x)
        weight = torch.softmax(torch.matmul(k.permute(0, 2, 1), q)*(v.shape[0])**(-0.5), dim = 1)
        out = torch.matmul(v, weight)
        out += self.bias

        # Normalization and Dropout after Nadaraya
        out = self.norm(out)
        out = self.dropout(out)

        # Nonlinear Output
        out = nn.GELU()(out)
        out = out.reshape(-1, self.n_seq*self.n_hid)

        # Variational Inference
        mu = self.out_layer_mu(out)
        log_var = self.out_layer_logvar(out)
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        out = mu + epsilon * std
        out = out.reshape(-1, self.n_seq, self.n_dim)
        return out, mu, log_var