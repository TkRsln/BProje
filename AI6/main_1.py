# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:01:26 2023

@author: utkua
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from moduls import MINT_Discriminator as Discriminator
from moduls import MINT_Generator as Generator

# Set random seed for reproducibility
torch.manual_seed(42)

# Display the generated samples
import matplotlib.pyplot as plt

def draw(output_images):
    img=output_images[0]#torch.concat((image_out,image_in),dim=2)
    img = img.view(28,28)
    #img = img.permute(1, 2, 0)
    img=img.detach().numpy()
    plt.imshow(img)
    
    # Display the image using PyPlot
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Remove axis labels
    plt.show()

def draw2(output_img):
    
    fig, axes = plt.subplots(1, 4, figsize=(8, 8))
    
    for i in range(4):
        axes[i].imshow(output_img[i].view(28, 28).detach().numpy(), cmap="gray")
        axes[i].axis("off")
        
    plt.show()

def load_params(train_id):
    global image_dim, label_dim, latent_dim
    model_d = Discriminator(image_dim, label_dim)
    model_d.load_state_dict(torch.load(f"./saves/discriminator_{str(train_id)}"))
    model_d.eval()
    model_g = Generator(latent_dim, label_dim, image_dim)
    model_g.load_state_dict(torch.load(f"./saves/generator_{str(train_id)}"))
    model_g.eval()
    return model_g,model_d

def save_params(train_id):
    torch.save(generator.state_dict(), f"./saves/generator_{str(train_id)}")
    torch.save(discriminator.state_dict(), f"./saves/discriminator_{str(train_id)}")


# Hyperparameters
latent_dim = 100
label_dim = 10
image_dim = 28 * 28
batch_size = 64
num_epochs = 50
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

create_new=True
if create_new:
    # Initialize the generator and discriminator
    generator = Generator(latent_dim, label_dim, image_dim).to(device)
    discriminator = Discriminator(image_dim, label_dim).to(device)
else:
    generator,discriminator=load_params(200)


# Define loss function and optimizers
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)


# Training loop
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.view(-1, image_dim).to(device)
        labels = torch.eye(label_dim)[labels].to(device)  # One-hot encoding
# ...
        # Train the discriminator
        discriminator_optimizer.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        real_validity = discriminator(real_images, labels)
        real_loss = adversarial_loss(real_validity, real_labels)

        noise = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(noise, labels)
        fake_validity = discriminator(fake_images, labels)
        fake_loss = adversarial_loss(fake_validity, fake_labels)

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        noise = torch.randn(real_images.size(0), latent_dim).to(device)
        fake_images = generator(noise, labels)
        fake_validity = discriminator(fake_images, labels)
        generator_loss = adversarial_loss(fake_validity, real_labels)

        generator_loss.backward()
        generator_optimizer.step()
        
        # Print training progress
        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                f"Generator Loss: {generator_loss.item():.4f}"
            )
        if (i+1) %10==0:
            draw2(fake_images)

# Generate some samples
num_samples = 10
sample_noise = torch.randn(num_samples, latent_dim).to(device)
sample_labels = torch.eye(label_dim).to(device)
generated_samples = generator(sample_noise, sample_labels).detach().cpu()

# Display the generated samples
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))

for i in range(num_samples):
    axes[i].imshow(generated_samples[i].view(28, 28), cmap="gray")
    axes[i].axis("off")

plt.show()




save_params(200)
