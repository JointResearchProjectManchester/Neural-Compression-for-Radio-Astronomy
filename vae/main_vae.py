import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import combinations, product
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal, kl_divergence, MultivariateNormal
from itertools import cycle  # (Unused; consider removing if not needed)
from datasets import RGZ108k
import wandb
import os

from encoder_vae import Encoder
from decoder_vae import Decoder
from residual import ResidualBlock, ResidualStack
from plotting_functions import denormalize, display_images


def find_correlation_matrix(image_size, sigma):
    """
    Create a pixel-to-pixel correlation matrix for a square image.
    Inputs:
      - image_size: height/width of the image
      - sigma: standard deviation used in the Gaussian correlation.
    """
    x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")
    pixel_coords = np.stack((x.ravel(), y.ravel()), axis=1)
    i, j = pixel_coords[:, 0], pixel_coords[:, 1]

    di = i[:, None] - i[None, :]  # Difference in x for all pairs
    dj = j[:, None] - j[None, :]  # Difference in y for all pairs
    d = 1.8 * np.sqrt(di**2 + dj**2)  # Scaled Euclidean distances

    C = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-d**2 / (2 * sigma**2))
    np.fill_diagonal(C, 1)  # Set diagonal to 1
    return C


# Training parameters
batch_size = 4
image_size = 150
num_training_updates = 150000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
latent_dim = 64  
learning_rate = 2e-4
sigma = 5.4 / (2 * np.sqrt(2 * np.log(2)))

# Custom dataset that loads from a memory-mapped numpy array
class MemoryMappedDataset(Dataset):
    def __init__(self, mmap_data, device):
        self.data = mmap_data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the numpy array to a torch tensor and move to the designated device.
        return torch.tensor(self.data[idx], dtype=torch.float32)


wandb.init(
    config={
        "latent_dim": latent_dim,
        "num_hiddens": num_hiddens,
        "num_residual_hiddens": num_residual_hiddens,
        "num_residual_layers": num_residual_layers,
        "architecture": "VAE",
        "num_training_updates": num_training_updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate        
    },
    dir='/share/nas2_3/adey/astro/wandb/'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed datasets
train_data = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_train_original.pth")
valid_data = torch.load("/share/nas2_3/amahmoud/week5/galaxy_out/processed_valid_original.pth")

# Save as numpy arrays (if not already saved)
np.save('/share/nas2_3/adey/astro/galaxy_out/valid_data_original.npy', np.stack([t.numpy() for t in valid_data]))
np.save('/share/nas2_3/adey/astro/galaxy_out/train_data.npy', np.stack([t.numpy() for t in train_data]))

# Load the datasets using memory mapping
train_data_mmap = np.load('/share/nas2_3/adey/astro/galaxy_out/train_data.npy', mmap_mode='r')
valid_data_mmap = np.load('/share/nas2_3/adey/astro/galaxy_out/valid_data_original.npy', mmap_mode='r')

# Create dataset objects
train_dataset = MemoryMappedDataset(train_data_mmap, device)
valid_dataset = MemoryMappedDataset(valid_data_mmap, device)

# Create DataLoaders (note: each sample is a single tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize the VAE model components
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim).to(device)
decoder = Decoder(latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=learning_rate,
)
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_updates, eta_min=1e-6)

# Lists for logging metrics
iterations = []
valid_iterations = []
train_total_loss = []
train_bits_per_dim = []
valid_total_loss = []
valid_bits_per_dim = []

encoder.train()
decoder.train()

# Precompute the correlation matrix and move it to GPU.
correlation_matrix = find_correlation_matrix(image_size, sigma)
correlation_matrix = torch.from_numpy(correlation_matrix).float().to(device)

print("Beginning training loop...")

i = 1
while i < num_training_updates:
    for images in train_loader:
        # If your images have an extra channel dimension (e.g., [B, 1, H, W]),
        # remove it if needed.
        images = images.to(device)
        images = images.squeeze(2)
        
        # --------------------
        # Forward pass (Training)
        # --------------------
        mean, logvar = encoder(images)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        x_recon = decoder(z)
        
        # Flatten outputs and targets
        x_recon_flat = x_recon.view(images.size(0), -1)
        images_flat = images.view(images.size(0), -1)

        # Compute reconstruction loss using a multivariate Gaussian likelihood on GPU.
        mvn = MultivariateNormal(loc=x_recon_flat, covariance_matrix=correlation_matrix)
        recon_loss = -mvn.log_prob(images_flat).sum()

        # KL Divergence between q(z|x) and the standard normal p(z)
        q_z_x = Normal(mean, torch.exp(0.5 * logvar))
        p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum() / mean.size(0)
        
        loss = recon_loss + kl_div

        # --------------------
        # Backpropagation & Optimization
        # --------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Bits per Dimension calculation
        bits_per_dim = recon_loss / (batch_size * image_size * image_size * np.log(2))

        wandb.log({
            "train/total_loss": loss.item(),
            "train/bits_per_dim": bits_per_dim.item()
        }, step=i)
        
        train_total_loss.append(loss.item())
        train_bits_per_dim.append(bits_per_dim.item())
        iterations.append(i)
        
        i += 1

        if i % 100 == 0:
            wandb.log({
                "train/total_loss_per_100": np.mean(train_total_loss[-100:]),
                "train/bits_per_dim_per_100": np.mean(train_bits_per_dim[-100:])
            }, step=i)
            
            print(f"{i} iterations, loss: {np.mean(train_total_loss[-100:]):.3f}, "
                  f"bits per dimension: {np.mean(train_bits_per_dim[-100:]):.5f}")
            
            # Clean up variables to help GPU memory management
            del images, mean, logvar, std, eps, z, x_recon, x_recon_flat, images_flat
            torch.cuda.empty_cache()

        # --------------------
        # Every 1000 iterations: perform validation
        # --------------------
        if i % 1000 == 0:
            encoder.eval()
            decoder.eval()
            
            valid_total_loss_batch = []
            valid_bits_per_dim_batch = []
            
            with torch.no_grad():
                for val_images in valid_loader:
                    val_images = val_images.to(device)
                    if val_images.dim() == 4 and val_images.size(1) == 1:
                        val_images = val_images.squeeze(1)
                    
                    mean, logvar = encoder(val_images)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mean + eps * std

                    x_recon = decoder(z)
                    x_recon_flat = x_recon.view(val_images.size(0), -1)
                    val_images_flat = val_images.view(val_images.size(0), -1)
                    
                    mvn = MultivariateNormal(loc=x_recon_flat, covariance_matrix=correlation_matrix)
                    recon_loss_val = -mvn.log_prob(val_images_flat).sum()
                    
                    q_z_x = Normal(mean, torch.exp(0.5 * logvar))
                    p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
                    kl_div_val = kl_divergence(q_z_x, p_z).sum() / mean.size(0)
                    
                    loss_val = recon_loss_val + kl_div_val
                    bits_per_dim_val = recon_loss_val / (val_images.size(0) * image_size * image_size * np.log(2))
                    
                    valid_total_loss_batch.append(loss_val.item())
                    valid_bits_per_dim_batch.append(bits_per_dim_val.item())
                    
                    del val_images, mean, logvar, std, eps, z, x_recon, x_recon_flat, val_images_flat
                    torch.cuda.empty_cache()
            
            avg_valid_loss = np.mean(valid_total_loss_batch)
            avg_valid_bits_per_dim = np.mean(valid_bits_per_dim_batch)
            
            wandb.log({
                "val/total_loss": avg_valid_loss,
                "val/bits_per_dim": avg_valid_bits_per_dim
            }, step=i)
            
            valid_total_loss.append(avg_valid_loss)
            valid_bits_per_dim.append(avg_valid_bits_per_dim)
            valid_iterations.append(i)
            
            print(f"Validation loss: {avg_valid_loss:.3f}, bits per dimension: {avg_valid_bits_per_dim:.5f}")
            
            encoder.train()
            decoder.train()
        
        if i >= num_training_updates:
            break


# Evaluation (Optional)
encoder.eval()
decoder.eval()

with torch.no_grad():
    for images in valid_loader:
        images = images.to(device)
        if images.dim() == 4 and images.size(1) == 1:
            images = images.squeeze(1)
            
        mean, logvar = encoder(images)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_recon = decoder(z)

        # Using MSE loss here for a quick evaluation; adjust as needed.
        nll = F.mse_loss(x_recon, images, reduction='sum')
        bits_per_dim_eval = nll / (images.size(0) * image_size * image_size * np.log(2))
        print('Bits per dimension (evaluation):', bits_per_dim_eval.item())
        break  # Remove break to evaluate on the entire validation set

# Generate and display images from a batch in the validation set
with torch.no_grad():
    images = next(iter(valid_loader))
    images = images.to(device)
    if images.dim() == 4 and images.size(1) == 1:
        images = images.squeeze(1)
    
    mean, logvar = encoder(images)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    x_recon = decoder(z)

display_images(original_images=images, reconstructed_images=x_recon, step=i)

# Save the model checkpoints
save_directory = '/share/nas2_3/adey/astro/galaxy_out/'
model_path = os.path.join(save_directory, 'vae_model.pth')
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
}, model_path)

model_artifact = wandb.Artifact('vae_model', type='model')
model_artifact.add_file(model_path)
wandb.log_artifact(model_artifact)

print("Model and training metrics saved successfully.")
