import torch
from model import VAE
from utils import make_perturbed_lattice, loss_function, write_xyz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("params.pt", weights_only=True))
mu = torch.load("mu.pt", weights_only=True)
sigma = torch.load("sigma.pt", weights_only=True)

num_samples = 300
#samples = torch.vstack([
#    make_perturbed_lattice(num_samples // 3) * 1.0,
#    make_perturbed_lattice(num_samples // 3) * 1.5,
#    make_perturbed_lattice(num_samples // 3) * 2.0,
#    ])
samples = make_perturbed_lattice(num_samples, global_scale=[1.0, 2.0])
samples_1d = samples.reshape(num_samples, -1)
sample_z = (samples_1d - mu) / sigma
x = sample_z.to(device)
#mu, logvar = model.encoder(x)
#recon = model.decoder(mu)
recon, z_mu, z_logvar = model(x)

loss, mse, kl = loss_function(recon, x, z_mu, z_logvar)
print(f"test loss: {loss.item()}")

x_hat = recon.to("cpu")
coord_1d = x_hat * sigma + mu
coord = coord_1d.reshape(num_samples, -1, 3)

write_xyz(coord, 'traj.xyz', elem='Cl')