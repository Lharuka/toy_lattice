import torch
import numpy as np
import torch.nn.functional as F

def make_perturbed_lattice(num_samples, x_range=(0, 5), y_range=(0, 5), z_range=(0, 5),
                             nx=6, ny=6, nz=6, noise_scale=0.1, global_scale=None):
    samples = []
    for _ in range(num_samples):
        x = torch.linspace(*x_range, nx)
        y = torch.linspace(*y_range, ny)
        z = torch.linspace(*z_range, nz)
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        lattice = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)

        # 施加扰动
        lattice += torch.normal(mean=0, std=noise_scale, size=lattice.shape)
        if global_scale is None:
            samples.append(lattice)
        else:
            r = np.random.uniform(*global_scale)
            samples.append(lattice * r)
    samples = torch.stack(samples)
    return samples

def loss_function(recon, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(recon, x)# * recon.shape[1]
    KL  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return MSE + beta * KL, MSE, KL

def write_xyz(x, filename, elem='C', comment=''):
    """
    x : ndarray, shape=(n_frame, N, 3)
    filename : 输出文件名，可带 .xyz 后缀
    elem : 元素符号，单原子体系只需一个字符串
    comment : 第二行注释
    """
    n_frame, N, _ = x.shape
    with open(filename, 'w') as f:
        for iframe, coord in enumerate(x):   # coord 形状 (N, 3)
            f.write(f'{N}\n')
            f.write(f'frame {iframe} {comment}\n')
            for r in coord:
                f.write(f'{elem:2s} {r[0]:15.8f} {r[1]:15.8f} {r[2]:15.8f}\n')