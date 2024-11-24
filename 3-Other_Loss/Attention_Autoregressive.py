import torch
import torch.nn as nn
from DTW import SoftDTW
from MMD import MMD

criterion1 = SoftDTW
criterion2 = MMD
class Attn_AR(nn.Module):
    def __init__(self, n_seq, n_hid, n_dim):
        super(Attn_AR, self).__init__()
        # Layer Design
        self.k_w = nn.Linear(n_dim, n_hid)
        self.q_w = nn.Linear(n_dim, n_hid)
        self.v_w = nn.Linear(n_dim, n_hid)
        self.bias = nn.Parameter(torch.randn(size = (n_seq, n_hid)))
        self.out_layer = nn.Linear(n_seq*n_hid, n_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(n_hid)

        nn.init.xavier_uniform_(self.k_w.weight)
        nn.init.xavier_uniform_(self.q_w.weight)
        nn.init.xavier_uniform_(self.v_w.weight)

        # Parameter Setup
        self.n_seq, self.n_hid, self.n_dim = n_seq, n_hid, n_dim

    def forward(self, x):
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
        out = self.out_layer(out).unsqueeze(1)
        return out