import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Normal, kl_divergence, MultivariateNormal
from datasets import RGZ108k
import wandb
import os

from encoder_vae import Encoder
from decoder_vae import Decoder
from residual import ResidualBlock, ResidualStack
from plotting_functions import denormalize, display_images

# Hyperparameters
batch_size = 4
image_size = 150
num_training_updates = 10000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
latent_dim = 64  
learning_rate = 2e-4

# Initialize wandb
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

# Define transforms and datasets
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

# Initialize the Encoder and Decoder models
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, latent_dim).to(device)
decoder = Decoder(latent_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(device)

# Define optimizer and scheduler
optimizer = optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=learning_rate,
)
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_updates, eta_min=1e-6)

# Tracking variables
iterations = []
valid_iterations = []
train_total_loss = []
train_bits_per_dim = []
valid_total_loss = []
valid_bits_per_dim = []

encoder.train()
decoder.train()

global_step = 0
epoch = 0

print("Beginning training loop...")
while global_step < num_training_updates:
    epoch += 1
    for images, _ in train_loader:
        if global_step >= num_training_updates:
            break

        images = images.to(device)
        mean, logvar = encoder(images)

        # Reparameterization Trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        """# Reconstruction Loss
        # reconstruction using a multivariate Gaussian likelihood:
        x_recon = decoder(z)
        # flatten both the reconstructed images and the targets for pixel-wise likelihood
        x_recon_flat = x_recon.view(images.size(0), -1)
        images_flat = images.view(images.size(0), -1)
        # create a diagonal (identity) covariance matrix (variance=1 for each pixel)
        scale = torch.ones_like(x_recon_flat)
        scale_tril = torch.diag_embed(scale)
        mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
        recon_loss = -mvn.log_prob(images_flat).sum()"""

        # traditional reconstruction loss (1D Gaussian/MSE):
        
        x_recon = decoder(z)
        recon_loss = F.mse_loss(x_recon, images, reduction='sum')
    

        # KL Divergence
        q_z_x = Normal(mean, torch.exp(0.5 * logvar))
        p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum() / mean.size(0)

        # Total Loss
        loss = recon_loss + kl_div

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute Bits per Dimension
        bits_per_dim = recon_loss / (batch_size * image_size * image_size * np.log(2))

        global_step += 1
        iterations.append(global_step)
        train_total_loss.append(loss.item())
        train_bits_per_dim.append(bits_per_dim.item())

        # Log Training Metrics Every 100 Updates
        if global_step % 100 == 0:
            wandb.log({
                "train/total_loss_per_100": np.mean(train_total_loss[-100:]),
                "train/bits_per_dim_per_100": np.mean(train_bits_per_dim[-100:])
            }, step=global_step)
            print(f"{global_step} iterations, loss: {np.mean(train_total_loss[-100:]):.3f}, "
                  f"bits per dimension: {np.mean(train_bits_per_dim[-100:]):.5f}")

        # Log individual training metrics
        wandb.log({
            "train/total_loss": loss.item(),
            "train/bits_per_dim": bits_per_dim.item()
        }, step=global_step)

        # Validation Every 1000 Updates
        if global_step % 1000 == 0:
            encoder.eval()
            decoder.eval()
            valid_total_loss_batch = []
            valid_bits_per_dim_batch = []

            with torch.no_grad():
                for images, _ in valid_loader:
                    images = images.to(device)
                    # Encoder forward pass on validation data
                    mean, logvar = encoder(images)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mean + eps * std

                    # Multivariate Gaussian reconstruction loss on validation data:
                    x_recon = decoder(z)
                    x_recon_flat = x_recon.view(images.size(0), -1)
                    images_flat = images.view(images.size(0), -1)
                    scale = torch.ones_like(x_recon_flat)
                    scale_tril = torch.diag_embed(scale)
                    mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
                    recon_loss = -mvn.log_prob(images_flat).sum()

                    # Alternative traditional reconstruction loss:
                    """
                    x_recon = decoder(z)
                    recon_loss = F.mse_loss(x_recon, images, reduction='sum')
                    """

                    q_z_x = Normal(mean, torch.exp(0.5 * logvar))
                    p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
                    kl_div = kl_divergence(q_z_x, p_z).sum() / mean.size(0)
                    loss_val = recon_loss + kl_div

                    bits_per_dim_val = recon_loss / (images.size(0) * image_size * image_size * np.log(2))

                    valid_total_loss_batch.append(loss_val.item())
                    valid_bits_per_dim_batch.append(bits_per_dim_val.item())

            avg_valid_loss = np.mean(valid_total_loss_batch)
            avg_valid_bits_per_dim = np.mean(valid_bits_per_dim_batch)

            wandb.log({
                "val/total_loss": avg_valid_loss,
                "val/bits_per_dim": avg_valid_bits_per_dim
            }, step=global_step)

            valid_total_loss.append(avg_valid_loss)
            valid_bits_per_dim.append(avg_valid_bits_per_dim)
            valid_iterations.append(global_step)
            print(f"Validation loss: {avg_valid_loss:.3f}, bits per dimension: {avg_valid_bits_per_dim:.5f}")

            encoder.train()
            decoder.train()

print("Training complete.")

# Evaluation (Optional)
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

        # Compute negative log likelihood via the multivariate Gaussian
        x_recon_flat = x_recon.view(images.size(0), -1)
        images_flat = images.view(images.size(0), -1)
        scale = torch.ones_like(x_recon_flat)
        scale_tril = torch.diag_embed(scale)
        mvn = MultivariateNormal(loc=x_recon_flat, scale_tril=scale_tril)
        nll = -mvn.log_prob(images_flat).sum()

        bits_per_dim_eval = nll / (images.size(0) * image_size * image_size * np.log(2))
        print('Evaluation Bits per dimension: ', bits_per_dim_eval.item())
        break  # Remove break to evaluate over the entire validation set

# Generate and Display Images
with torch.no_grad():
    images, _ = next(iter(valid_loader))
    images = images.to(device)
    mean, logvar = encoder(images)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mean + eps * std
    x_recon = decoder(z)
    display_images(original_images=images, reconstructed_images=x_recon, step=global_step)

# Save the Model
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
