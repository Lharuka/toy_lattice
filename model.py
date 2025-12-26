import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(512), nn.Tanh(),
            nn.LazyLinear(256), nn.Tanh(),
            nn.LazyLinear(128), nn.Tanh(),
        )

        self.mean = nn.LazyLinear(100)
        self.logv = nn.LazyLinear(100)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.logv(h)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyLinear(128), nn.Tanh(),
            nn.LazyLinear(256), nn.Tanh(),
            nn.LazyLinear(512), nn.Tanh(),
            nn.LazyLinear(648)
        )

    def forward(self, z):
        return self.net(z)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.rand_like(std)
    return mu + eps * std

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar