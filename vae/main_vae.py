import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import combinations, product
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal, kl_divergence,MultivariateNormal
from itertools import cycle
from datasets import RGZ108k
import wandb
import os

from encoder_vae import Encoder
from decoder_vae import Decoder
from residual import ResidualBlock,ResidualStack
from plotting_functions import denormalize, display_images


batch_size = 4
image_size = 150
num_training_updates = 125000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
latent_dim = 64  
learning_rate = 2e-4

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

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.0031,), (0.0352,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "/share/nas2_3/adey/data/galaxy_zoo/"
train_dataset = RGZ108k(root=dataset_path, train=True, transform=transform, target_transform=None, download=False)
valid_dataset = RGZ108k(root=dataset_path, train=False, transform=transform, target_transform=None, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim).to(device)
decoder = Decoder(latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=learning_rate,
)

scheduler = CosineAnnealingLR(optimizer, T_max=num_training_updates, eta_min=1e-6)

iterations = []
valid_iterations = []
train_total_loss = []
train_bits_per_dim = []
valid_total_loss = []
valid_bits_per_dim = []
encoder.train()
decoder.train()


train_loader_iter = cycle(train_loader)

print("Beginning training loop...")
for i in range(1, num_training_updates + 1):
    # Get a batch from the training data loader
    images, _ = next(train_loader_iter)
    images = images.to(device)
    
    #  Encoder forward pass 
    mean, logvar = encoder(images)
    
    #  Reparameterization Trick 
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std

    #  Reconstruction Loss 
    # Traditional reconstruction loss using a 1D Gaussian (MSE)
    """
    x_recon = decoder(z)
    recon_loss = F.mse_loss(x_recon, images, reduction='sum')
    """
    
    # Reconstruction loss using a multivariate Gaussian likelihood:
    x_recon = decoder(z)
    # Flatten images to a vector per sample (pixel-wise likelihood)
    x_recon_flat = x_recon.view(images.size(0), -1)
    images_flat = images.view(images.size(0), -1)
    
    # Determine the dimensionality (number of pixels)
    D = image_size*image_size 

    # Create a diagonal (identity) covariance matrix with variance=1 for each sample.
    scale = torch.ones_like(x_recon_flat)   
    scale_tril = torch.diag_embed(scale)   
    
    # Define a multivariate normal distribution for each sample in the batch.
    mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
    
    # Compute the negative log likelihood (NLL) loss (summed over the batch)
    recon_loss = -mvn.log_prob(images_flat).sum()

    #  KL Divergence 
    # Using the helper function _loss (which computes the average KL divergence per batch)
    q_z_x = Normal(mean, torch.exp(0.5 * logvar))
    p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
    kl_div = kl_divergence(q_z_x, p_z).sum() / mean.size(0)

    #  Total Loss 
    loss = recon_loss + kl_div

    #  Backpropagation and Optimization 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    #  Compute Bits per Dimension 
    bits_per_dim = recon_loss / (batch_size * image_size * image_size * np.log(2))
    
    #  Track Metrics 
    train_total_loss.append(loss.item())
    train_bits_per_dim.append(bits_per_dim.item())
    iterations.append(i)
    
    #  Log Training Metrics Every 100 Iterations 
    if i % 100 == 0:
        wandb.log({
            "train/total_loss_per_100": np.mean(train_total_loss[-100:]),
            "train/bits_per_dim_per_100": np.mean(train_bits_per_dim[-100:])
        }, step=i)
        
        print(f"{i} iterations, loss: {np.mean(train_total_loss[-100:]):.3f}, "
              f"bits per dimension: {np.mean(train_bits_per_dim[-100:]):.5f}")
    
    # Log individual training metrics
    wandb.log({
        "train/total_loss": loss.item(),
        "train/bits_per_dim": bits_per_dim.item()
    }, step=i)
    
    #  Validation Every 1000 Iterations 
    if i % 1000 == 0:
        encoder.eval()
        decoder.eval()
        
        valid_total_loss_batch = []
        valid_bits_per_dim_batch = []
        
        with torch.no_grad():
            for images, _ in valid_loader:
                images = images.to(device)
                
                # Encoder forward pass
                mean, logvar = encoder(images)
                
                # Reparameterization
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std
                
                # Traditional reconstruction loss (MSE) alternative:
                """
                x_recon = decoder(z)
                recon_loss = F.mse_loss(x_recon, images, reduction='sum')
                """
                
                # Multivariate Gaussian reconstruction loss:
                x_recon = decoder(z)
                x_recon_flat = x_recon.view(images.size(0), -1)
                images_flat = images.view(images.size(0), -1)
                
                # Use the same D and construct the diagonal covariance
                D = x_recon_flat.size(1)
                scale = torch.ones_like(x_recon_flat)
                scale_tril = torch.diag_embed(scale)
                
                mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
                recon_loss = -mvn.log_prob(images_flat).sum()
                
                # Compute KL divergence
                q_z_x = Normal(mean, torch.exp(0.5 * logvar))
                p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
                kl_div = kl_divergence(q_z_x, p_z).sum() / mean.size(0)
                
                # Total loss
                loss = recon_loss + kl_div
                
                # Bits per dimension for this batch
                bits_per_dim = recon_loss / (images.size(0) * image_size * image_size * np.log(2))
                
                valid_total_loss_batch.append(loss.item())
                valid_bits_per_dim_batch.append(bits_per_dim.item())
        
        # Compute average validation metrics over the validation set
        avg_valid_loss = np.mean(valid_total_loss_batch)
        avg_valid_bits_per_dim = np.mean(valid_bits_per_dim_batch)
        
        # Log validation metrics to wandb
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

#  Evaluation (Optional) 
encoder.eval()
decoder.eval()


with torch.no_grad():
    for images, _ in valid_loader:
        images = images.to(device)
        mean, logvar = encoder(images)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_recon = decoder(z)

        nll = F.mse_loss(x_recon, images, reduction='sum')  
        bits_per_dim = nll / (images.size(0) * image_size * image_size * np.log(2))  

        print('Bits per dimension: ', bits_per_dim.item()) 
        break  # Remove to compute over the entire validation set

# Generate and display images
with torch.no_grad():
    images, _ = next(iter(valid_loader)) 
    images = images.to(device)
    
    mean, logvar = encoder(images)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    x_recon = decoder(z)

display_images(original_images=images, reconstructed_images=x_recon, step=i)

# Save the Encoder and Decoder
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
