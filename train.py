import torch
from torch.utils.data import TensorDataset, DataLoader
from model import VAE
from utils import make_perturbed_lattice, loss_function
import matplotlib.pyplot as plt
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

num_train = cfg["data"]["num_train"]
num_valid = cfg["data"]["num_valid"]
scale = cfg["data"]["global_scale"]
train_data = make_perturbed_lattice(num_train, global_scale=scale)
valid_data = make_perturbed_lattice(num_valid, global_scale=scale)
train_data_1d = train_data.reshape(num_train, -1)
valid_data_1d = valid_data.reshape(num_valid, -1)

mu = train_data_1d.mean(dim=0)
sigma = train_data_1d.std(dim=0)
torch.save(mu, "mu.pt")
torch.save(sigma, "sigma.pt")

train_data_z = (train_data_1d - mu) / sigma
valid_data_z = (valid_data_1d - mu) / sigma

batch_size = cfg["train"]["batch_size"]
train_set = TensorDataset(train_data_z)
valid_set = TensorDataset(valid_data_z)
train_loader = DataLoader(train_set, batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size, shuffle=False)

model = VAE()
lr = cfg["train"]["lr"]
num_epoch = cfg["train"]["num_epoch"]
beta = cfg["train"]["beta"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
if cfg["train"]["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
best_loss = 114514

train_hist = []
valid_hist = []
for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    for x in train_loader:
        x = x[0].to(device)
        recon, mu, logvar = model(x)
        loss, mse, kl = loss_function(recon, x, mu, logvar, beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x)

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for x in valid_loader:
            x = x[0].to(device)
            recon, mu, logvar = model(x)
            loss, mse, kl = loss_function(recon, x, mu, logvar)
            valid_loss += loss.item() * len(x)

    train_loss /= num_train
    valid_loss /= num_valid
    train_hist.append(train_loss)
    valid_hist.append(valid_loss)
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_param = model.state_dict()
        best_epoch = epoch + 1
    print(f"epoch: {epoch + 1}, train loss: {train_loss}, valid loss: {valid_loss}")
print(f"best epoch: {best_epoch}, best loss: {best_loss}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
torch.save(best_param, "params.pt")

epochs = [epoch + 1 for epoch in range(num_epoch)]

plt.figure()
plt.plot(epochs, train_hist, label="train")
plt.plot(epochs, valid_hist, label="valid")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
if len(epochs) >= 1000:
    plt.xscale("log")
plt.savefig("loss.png")