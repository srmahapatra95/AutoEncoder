# Importing all the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()


class DownsamplingEncoder(nn.Module):

    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),


        )

    def forward(self, x):
        return self.encoder(x)
    

def save_image_grid(x_tensor, filename, nrow=8):
    """Save a grid of images"""
    grid = utils.make_grid(x_tensor.cpu(), nrow=nrow, padding=2)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(ndarr)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved image grid to {filename}")

    

class UpSamplingDecoder(nn.Module):
    def __init__(self, latent_dim = 128, out_channels = 3):
        super().__init__()

        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 14 -> 28
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 28 -> 56
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 56 -> 112
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 112 -> 224
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class AutoEncoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=128, bottleneck_dim=16):
        super().__init__()
        self.encoder = DownsamplingEncoder(in_channels, latent_dim)
        self.latent_dim = latent_dim

        # True bottleneck - flatten spatial features and compress to bottleneck_dim
        # After encoder: (B, latent_dim, 7, 7) for 224x224 input
        self.fc_encode = nn.Linear(latent_dim * 7 * 7, bottleneck_dim)
        self.fc_decode = nn.Linear(bottleneck_dim, latent_dim * 7 * 7)

        self.decoder = UpSamplingDecoder(latent_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)  # Shape: (B, latent_dim, 7, 7)

        # Flatten and compress through bottleneck
        z_flat = z.view(z.size(0), -1)  # (B, latent_dim * 49)
        z_bottleneck = self.fc_encode(z_flat)  # (B, bottleneck_dim) - TRUE bottleneck

        # Decode from bottleneck
        z_decoded = self.fc_decode(z_bottleneck)  # (B, latent_dim * 49)
        z_reshaped = z_decoded.view(-1, self.latent_dim, 7, 7)  # (B, latent_dim, 7, 7)

        recon = self.decoder(z_reshaped)
        return recon, z_bottleneck
    

def train_autoencoder(model, train_loader, test_loader, num_epochs=30, lr = 1e-3, device=DEVICE, save_dir='./checkpoints', model_name="Autoencoder"):
    """
    Docstring for train_one_epoch
    
    :param model: Description
    :param train_loader: Description
    :param test_loader: Description
    :param num_epochs: Description
    :param lr: Description
    :param device: Description
    :param save_dir: Description
    :param model_name: Description
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Training model name : {model_name}")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()

            recon, z = model(xb)
            loss = criterion(recon, xb)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(device)
                recon, z = model(xb)
                loss = criterion(recon, xb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(test_loader.dataset)

        print(f"[Epoch {epoch:03d}] train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))

        # Save reconstruction samples
        if epoch % 5 == 0:
            model.eval()
            xb, _ = next(iter(test_loader))
            xb = xb.to(device)
            with torch.no_grad():
                recon, _ = model(xb)
            both = torch.cat([xb.cpu(), recon.cpu()], dim=0)
            save_image_grid(both, os.path.join(save_dir, f'{model_name}_recon_epoch{epoch:03d}.png'), nrow=8)

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")
    return model


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    LATENT_DIM = 256
    BOTTLENECK_DIM = 128
    IMAGE_SIZE = 224
    IN_CHANNELS = 3
    DATA_DIR = "../data/archive/images"
    SAVE_DIR = "./checkpoints"
    TRAIN_SPLIT = 0.8

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Load dataset from archive/images folder
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=transform
    )

    # Split into train and test
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"Total images: {len(full_dataset)}")
    print(f"Training images: {train_size}, Test images: {test_size}")

    # pin_memory only works with CUDA, not MPS
    use_pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory
    )

    # Initialize model
    model = AutoEncoder(
        in_channels=IN_CHANNELS,
        latent_dim=LATENT_DIM,
        bottleneck_dim=BOTTLENECK_DIM
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE,
        save_dir=SAVE_DIR,
        model_name="AutoEncoder_Paintings"
    )


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    main()
