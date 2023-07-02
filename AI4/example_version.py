# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:46:13 2023

@author: utkua
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the generator network with an autoencoder
class Generator(nn.Module):
    def __init__(self, text_dim, image_dim, noise_dim):
        super(Generator, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.noise_dim = noise_dim

        self.text_embedding = nn.Embedding(text_dim, 128)
        self.encoder = nn.Sequential(
            nn.Linear(image_dim + 128 + noise_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, text_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(text_dim + 128 + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, text_labels, image, noise):
        text_embedded = self.text_embedding(text_labels).squeeze(1)
        x = torch.cat((text_embedded, image.view(-1, self.image_dim), noise), dim=1)
        encoded = self.encoder(x)
        generated_image = self.decoder(torch.cat((encoded, text_embedded, noise), dim=1))
        return generated_image.view(-1, 3, 900, 850)  # Reshape to RGB image

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, text_dim, image_dim):
        super(Discriminator, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim

        self.text_embedding = nn.Embedding(text_dim, 128)
        self.model = nn.Sequential(
            nn.Linear(image_dim + 128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, text_labels, image):
        text_embedded = self.text_embedding(text_labels).squeeze(1)
        x = torch.cat((text_embedded, image.view(-1, self.image_dim)), dim=1)
        validity = self.model(x)
        return validity

# Hyperparameters
text_dim = 10
image_dim = 3 * 900 * 850  # Image size with RGB channels
noise_dim = 100
batch_size = 64
num_epochs = 10
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset and apply transformations
transform = transforms.Compose([
    transforms.Resize((900, 850)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

# Initialize the generator and discriminator
generator = Generator(text_dim, image_dim, noise_dim).to(device)
discriminator = Discriminator(text_dim, image_dim, noise_dim).to(device)

# Define loss function and optimizers
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_images = real_images.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        
        # Generate random noise
        noise = torch.randn(batch_size, noise_dim).to(device)

        # Generate random text labels
        text_labels = torch.randint(0, text_dim, (batch_size, 1)).to(device)

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        real_validity = discriminator(text_labels, real_images)
        real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))
        
        # Generate fake images
        generated_images = generator(text_labels, real_images, noise)

        # Calculate discriminator loss for fake images
        fake_validity = discriminator(text_labels, generated_images.detach())
        fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

        # Total discriminator loss
        discriminator_loss = (real_loss + fake_loss) / 2.0
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        fake_validity = discriminator(text_labels, generated_images)
        generator_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))
        generator_loss.backward()
        generator_optimizer.step()

        # Print training progress
        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(dataloader)}], "
                f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                f"Generator Loss: {generator_loss.item():.4f}"
            )

