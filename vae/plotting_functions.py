import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 1, 1, 1).to(tensor.device)
    std = torch.tensor(std).reshape(1, 1, 1, 1).to(tensor.device)
    return tensor * std + mean

def display_images(original_images, reconstructed_images, num_images=16, step=0):
    # Adjust the number of images to display if the batch size is smaller than num_images
    num_images = min(num_images, original_images.shape[0])

    # Denormalize images (for grayscale, single mean and std)
    original_images = denormalize(original_images, (0.0031,), (0.0352,))
    reconstructed_images = denormalize(reconstructed_images, (0.0031,), (0.0352,))

    # Convert images to numpy and move channel dimension to last position
    original_images = original_images.cpu().numpy().transpose(0, 2, 3, 1)
    reconstructed_images = reconstructed_images.cpu().numpy().transpose(0, 2, 3, 1)

    # Force squeeze last dimension, assuming grayscale images
    original_images = original_images.squeeze(-1)
    reconstructed_images = reconstructed_images.squeeze(-1)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.25, 4))

    # Plot each pair of original and reconstructed images
    for i in range(num_images):
        # Display original images
        axes[0, i].imshow((original_images[i] * 255).astype(np.uint8), cmap='gray')
        axes[0, i].axis('off')

        # Display reconstructed images
        axes[1, i].imshow((reconstructed_images[i] * 255).astype(np.uint8), cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    save_directory = '/share/nas2_3/adey/astro/galaxy_out/reconstructed_vae.pdf'
    plt.savefig(save_directory)

    # Log the figure with wandb
    wandb.log({"Reconstructed Images": wandb.Image(fig)}, step=step)

    plt.show()
    plt.close(fig)