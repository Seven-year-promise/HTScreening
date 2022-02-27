import argparse
import torch
import torch.nn as nn
import pyro


class NN(nn.Module):
    def __init__(self, input_dim = 15, z_dim=2, classes=10):
        super(NN, self).__init__()

        # encoder
        self.enc1 = nn.Linear(input_dim, 256)
        self.enc2 = nn.Linear(256, 128)
        self.enc3 = nn.Linear(128, z_dim)
        self.classifier = nn.Linear(z_dim, classes)

    def get_latent(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        z = nn.functional.relu(self.enc3(x))

        return z

    def forward(self, x):
        # encoding
        x = nn.functional.relu(self.enc1(x))
        x = nn.functional.relu(self.enc2(x))
        z = nn.functional.relu(self.enc3(x))
        clasification = self.classifier(z)
        return clasification


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