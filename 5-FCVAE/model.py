import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class attention(nn.Module):
    def __init__(self, n_dim):
        super(attention, self).__init__()
        self.k_w = nn.Linear(n_dim, n_dim)
        self.q_w = nn.Linear(n_dim, n_dim)
        self.v_w = nn.Linear(n_dim, n_dim)
        self.scale = n_dim**(-.5)
    def forward(self, x):
        x = x.permute(0, 2, 1) #x:(N, W, 1) -> (N, 1, W)
        k = self.k_w(x) #k: (N, 1, W)
        q = self.q_w(x) #q: (N, 1, W)
        v = self.v_w(x) #v: (N, 1, W)
        weight = torch.softmax(torch.matmul(k.permute(0, 2, 1), q)*(self.scale), dim = 1) #k.permute(0, 2, 1):(N, W, 1), q:(N, 1, W) -> (N, W, W)
        out = torch.matmul(v, weight)
        return out

class GFM(nn.Module):
    def __init__(self, n_dim, n_hid):
        super(GFM, self).__init__()
        self.dense = nn.Linear(n_dim, n_hid) # (N, W) -> (N, d)
    def forward(self, x):
        out = torch.fft.fft(x).real # (N, W) -> (N, W)
        out = nn.ReLU()(self.dense(out)) # (N, W) -> (N, d)
        out = nn.Dropout(p=.1)(out) # (N, d)
        return out

class LFM(nn.Module):
    def __init__(self, n_dim, n_hid1, n_hid2, n_seq):
        super(LFM, self).__init__()
        self.dense = nn.Linear(n_seq, n_hid1)
        self.attn = attention(n_hid1)
        self.ffd = nn.Linear((n_dim//n_seq)*n_hid1, n_hid2)

        # parameters setup
        self.n_dim, self.n_seq, self.n_hid1, self.n_hid2 = n_dim, n_seq, n_hid1, n_hid2
    def forward(self, x):
        out = x.reshape(x.shape[0], self.n_dim//self.n_seq, self.n_seq) # (N, W) -> (N, n, k)
        out = torch.fft.fft(out).real # (N, n, k)
        out = self.dense(out).reshape(out.shape[0], self.n_hid1, -1) # (N, n, k) -> (N, n, l) -> (N, l, n)
        out = self.attn(out).reshape(out.shape[0], -1) # (N, l, n) -> (N, l, n) -> (N, n*l)
        out = nn.ReLU()(self.ffd(out)) # (N, n*l) -> (N, d)
        return out

class VAE(nn.Module):
    def __init__(self, n_hid2, n_dim):
        n_hid = n_hid2*2 + n_dim
        super(VAE, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(n_hid, n_hid//2),
            nn.GELU(),
            nn.Linear(n_hid//2, n_hid//4),
            nn.GELU(),
        )
        # Variational Inference
        self.mean = nn.Linear(n_hid//4, n_hid//8)
        self.logvar = nn.Linear(n_hid//4, n_hid//8)

        # Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(n_hid//8 + n_hid2*2, n_hid//4),
            nn.GELU(),
            nn.Linear(n_hid//4, n_hid//2),
            nn.GELU(),
        )

        # Variational Inference
        self.recon_mean = nn.Linear(n_hid//2, n_dim)
        self.recon_logvar = nn.Linear(n_hid//2, n_dim)

    def forward(self, x, LF, GF):
        # Encoder
        out = self.Encoder(x)
        
        # Variational Inference
        z_mean = self.mean(out)
        z_logvar = self.logvar(out)
        std = torch.exp(0.5*z_logvar)
        z = torch.randn_like(std)*std + z_mean

        # Decoder
        z = torch.cat((z, LF, GF), dim = 1)
        out = self.Decoder(z)

        # Variational Inference
        mean = self.recon_mean(out)
        logvar = self.recon_logvar(out)
        std = torch.exp(0.5*logvar)
        x_hat = torch.randn_like(std)*std + mean
        return x_hat

    def train(self, x, LF, GF):
        # Encoder
        out = self.Encoder(x)
        
        # Variational Inference
        z_mean = self.mean(out)
        z_logvar = self.logvar(out)
        std = torch.exp(0.5*z_logvar)
        z = torch.randn_like(std)*std + z_mean

        # Decoder
        z = torch.cat((z, LF, GF), dim = 1)
        out = self.Decoder(z)

        # Variational Inference
        mean = self.recon_mean(out)
        logvar = self.recon_logvar(out)
        return mean, logvar, z_mean, z_logvar

class FCVAE(nn.Module):
    def __init__(self, n_dim, n_hid1, n_hid2, n_seq):
        super(FCVAE, self).__init__()
        # Parameter Setup
        self.n_dim, self.n_hid1, self.n_hid2, self.n_seq = n_dim, n_hid1, n_hid2, n_seq

        # Layer Setup
        self.GFM = GFM(n_dim, n_hid2)
        self.LFM = LFM(n_dim, n_hid1, n_hid2, n_seq)
        self.VAE = VAE(n_hid2, n_dim)
    def forward(self, x):
        gf = self.GFM(x)
        lf = self.LFM(x)
        out = torch.cat((x, gf, lf), dim = 1)
        return self.VAE(out, gf, lf)

    def train(self, x):
        gf = self.GFM(x)
        lf = self.LFM(x)
        out = torch.cat((x, gf, lf), dim = 1)
        mean, logvar, z_mean, z_logvar = self.VAE.train(out, gf, lf)
        return mean, logvar, z_mean, z_logvar