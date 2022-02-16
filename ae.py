import argparse
import torch
import torch.nn as nn
import pyro
import pyro.contrib.gp as gp
from utils.kernels import LinearARD



class AE(nn.Module):
    def __init__(self, input_dim = 15, h_dim = 8, z_dim=2):
        super(AE, self).__init__()

        # encoder
        self.enc1 = nn.Linear(input_dim, input_dim)
        self.enc2 = nn.Linear(input_dim, input_dim)
        self.enc3 = nn.Linear(input_dim, h_dim)
        self.enc4 = nn.Linear(h_dim, h_dim)

        # fully connected layers for learning representations
        self.fc_mu = nn.Linear(h_dim, z_dim)
        # decoder
        self.dec1 = nn.Linear(z_dim, h_dim)
        self.dec2 = nn.Linear(h_dim, h_dim)
        self.dec3 = nn.Linear(h_dim, input_dim)
        self.dec4 = nn.Linear(input_dim, input_dim)

    def encoder(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        mu = self.fc_mu(x)

        return mu, None

    def get_latent(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        mu = self.fc_mu(x)

        return mu

    def forward(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        z = self.fc_mu(x)
        # get the latent vector through reparameterization

        # decoding
        x = nn.functional.relu(self.dec1(z))
        x = nn.functional.relu(self.dec2(x))
        x = nn.functional.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction

class AE_class(nn.Module):
    def __init__(self, input_dim = 15, h_dim = 8, z_dim=2, classes=10):
        super(AE_class, self).__init__()

        # encoder
        self.enc1 = nn.Linear(input_dim, input_dim)
        self.enc2 = nn.Linear(input_dim, input_dim)
        self.enc3 = nn.Linear(input_dim, h_dim)
        self.enc4 = nn.Linear(h_dim, h_dim)

        # fully connected layers for learning representations
        self.fc_mu = nn.Linear(h_dim, z_dim)
        # decoder
        self.dec1 = nn.Linear(z_dim, h_dim)
        self.dec2 = nn.Linear(h_dim, h_dim)
        self.dec3 = nn.Linear(h_dim, input_dim)
        self.dec4 = nn.Linear(input_dim, input_dim)

        self.fc_classifier = nn.Linear(z_dim, z_dim)
        self.classifier = nn.Linear(z_dim, classes)

    def encoder(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        mu = self.fc_mu(x)

        return mu, None

    def get_latent(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        mu = self.fc_mu(x)

        return mu

    def forward(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        # get `mu` and `log_var`
        z = self.fc_mu(x)

        # decoding
        x = nn.functional.relu(self.dec1(z))
        x = nn.functional.relu(self.dec2(x))
        x = nn.functional.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))

        class_fc = nn.functional.relu(self.fc_classifier(z))
        clasification = self.classifier(class_fc)
        return reconstruction, clasification

class ConvAE(nn.Module):
    def __init__(self, kernel_size=4, init_channels=8, image_channels=1, z_dim=16, h_dim=128):
        super(ConvAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2, out_channels=init_channels * 4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=64, kernel_size=kernel_size,
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, 64)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels * 8, kernel_size=kernel_size,
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8, out_channels=init_channels * 4, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4, out_channels=init_channels * 2, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2, out_channels=image_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def encoder(self, x):
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)

        return mu, None

    def decoder(self, z):
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = nn.functional.relu(self.dec1(z))
        x = nn.functional.relu(self.dec2(x))
        x = nn.functional.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction

    def forward(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        x = nn.functional.relu(self.enc3(x))
        x = nn.functional.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        # get the latent vector through reparameterization
        z = self.fc2(mu)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = nn.functional.relu(self.dec1(z))
        x = nn.functional.relu(self.dec2(x))
        x = nn.functional.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return mu, reconstruction

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=11, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-tf",
        "--test-frequency",
        default=10,
        type=int,
        help="how often we evaluate the test set",
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--jit", action="store_true", default=False, help="whether to use PyTorch jit"
    )
    parser.add_argument(
        "-visdom",
        "--visdom_flag",
        default=True,
        action="store_true",
        help="Whether plotting in visdom is desired",
    )
    parser.add_argument(
        "-i-tsne",
        "--tsne_iter",
        default=10,
        type=int,
        help="epoch when tsne visualization runs",
    )
    parser.add_argument(
        "--main_path",
        default="./vae_results_cifar10/VAE_CNN_rbf_torch_cifar10/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)