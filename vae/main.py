import os
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.distributions import Normal, kl_divergence, MultivariateNormal
from datasets import RGZ108k
from encoders import Encoder
from decoders import AEDecoder, VAEDecoder
from vectorquantizer import VectorQuantizer
from base_models import BaseAutoencoder
import plotting_functions

# Configuration
config = {
    "architecture": "AE",  # Can be 'AE', 'VAE', or 'VQ-VAE'
    "batch_size": 4,
    "image_size": 150,
    "num_training_updates": 10000,
    "learning_rate": 2e-4,
    "latent_dim": 512,
    "num_hiddens": 128,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "num_embeddings": 256,
    "commitment_cost": 0.25,
    "dataset_path": "/share/nas2_3/adey/data/galaxy_zoo/",
    "wandb_dir": "/share/nas2_3/adey/astro/wandb/",
    "save_dir": "/share/nas2_3/adey/astro/galaxy_out/",
    "save_latents_path": "/share/nas2_3/adey/astro/latents/"
}

wandb.init(config=config, project="comparing_models", entity="deya-03-the-university-of-manchester")

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.0031,), (0.0352,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = RGZ108k(root=config["dataset_path"], train=True, transform=transform)
valid_dataset = RGZ108k(root=config["dataset_path"], train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

# Initialize Model
encoder = Encoder(config["num_hiddens"], config["num_residual_layers"], config["num_residual_hiddens"], config["latent_dim"], variant=config["architecture"])

vq_layer = None
if config["architecture"] == "VQ-VAE":
    vq_layer = VectorQuantizer(config["num_embeddings"], config["latent_dim"], config["commitment_cost"])

decoder_type = config["architecture"]
if decoder_type == "AE":
    decoder = AEDecoder(latent_dim=config["latent_dim"], num_hiddens=config["num_hiddens"], num_residual_layers=config["num_residual_layers"], num_residual_hiddens=config["num_residual_hiddens"])
elif decoder_type in ["VAE", "VQ-VAE"]:
    decoder = VAEDecoder(latent_dim=config["latent_dim"], num_hiddens=config["num_hiddens"], num_residual_layers=config["num_residual_layers"], num_residual_hiddens=config["num_residual_hiddens"])

model = BaseAutoencoder(encoder, decoder_type=decoder_type, vq_layer=vq_layer, latent_dim=config["latent_dim"], num_hiddens=config["num_hiddens"], num_residual_layers=config["num_residual_layers"], num_residual_hiddens=config["num_residual_hiddens"]).to(device)

# Optimizer and Scheduler
optimizer = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=config["learning_rate"])
scheduler = CosineAnnealingLR(optimizer, T_max=config["num_training_updates"], eta_min=1e-6)

# Training Loop
model.train()
global_step = 0
print("starting training")
for epoch in range(10):
    for batch_idx, (images, _) in enumerate(train_loader):
        if global_step >= config["num_training_updates"]:
            break

        images = images.to(device)
        reconstructed, vq_loss = model(images)
        
        if config["architecture"] == "VAE":
            mean, logvar = model.encoder(images)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            recon_loss = F.mse_loss(model.decoder(z), images, reduction='sum')
            kl_div = kl_divergence(Normal(mean, std), Normal(torch.zeros_like(mean), torch.ones_like(std))).sum()
            total_loss = recon_loss + kl_div
        elif config["architecture"] == "VQ-VAE":
            z = model.encoder(images)
            quantized, vq_loss, _ = vq_layer(z)
            recon_loss = F.mse_loss(model.decoder(quantized), images, reduction='sum')
            total_loss = recon_loss + vq_loss
        else:  # AE
            recon_loss = F.mse_loss(reconstructed, images, reduction='sum')
            total_loss = recon_loss

        # Compute bits per dimension (bpd)
        bpd = recon_loss / (images.size(0) * config["image_size"] * config["image_size"] * np.log(2))
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        wandb.log({"train/loss": total_loss.item(), "train/recon_loss": recon_loss.item()}, step=global_step)
        wandb.log({"train/bpd": bpd.item()}, step=global_step)
        if config["architecture"] == "VQ-VAE":
            wandb.log({"train/vq_loss": vq_loss.item()}, step=global_step)
        if config["architecture"] == "VAE":
            wandb.log({"train/kl_div": kl_div.item()}, step=global_step)

        if global_step % 100 == 0:
            print(f"train/loss: {total_loss.item()},train/bpd: {bpd.item()}")

        
        if global_step % 1000 == 0:
            model.eval()
            val_total_loss, val_recon_loss, val_vq_loss, val_kl_div = [], [], [], []
            with torch.no_grad():
                for val_images, _ in valid_loader:
                    val_images = val_images.to(device)
                    val_recon, _ = model(val_images)
                    if config["architecture"] == "VAE":
                        mean, logvar = model.encoder(val_images)
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mean + eps * std
                        recon_loss = F.mse_loss(model.decoder(z), val_images, reduction='sum')
                        kl_div = kl_divergence(Normal(mean, std), Normal(torch.zeros_like(mean), torch.ones_like(std))).sum()
                        total_loss = recon_loss + kl_div
                        val_kl_div.append(kl_div.item())
                    elif config["architecture"] == "VQ-VAE":
                        z = model.encoder(val_images)
                        quantized, vq_loss, _ = vq_layer(z)
                        recon_loss = F.mse_loss(model.decoder(quantized), val_images, reduction='sum')
                        total_loss = recon_loss + vq_loss
                        val_vq_loss.append(vq_loss.item())
                    else:  # AE
                        recon_loss = F.mse_loss(val_recon, val_images, reduction='sum')
                        total_loss = recon_loss
                    
                    val_recon_loss.append(recon_loss.item())
                    val_total_loss.append(total_loss.item())
            avg_val_loss = np.mean(val_total_loss)
            avg_val_recon_loss = np.mean(val_recon_loss)
            avg_val_bpd = np.mean(val_bpd_list)
            wandb.log({"val/bpd": avg_val_bpd}, step=global_step)
            wandb.log({"val/loss": avg_val_loss, "val/recon_loss": avg_val_recon_loss}, step=global_step)

            model.train()
        
        global_step += 1

torch.save(model.state_dict(), os.path.join(config["save_dir"], f"{config["architecture"]}_model.pth"))

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
