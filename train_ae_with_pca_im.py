import argparse
import numpy as np
import torch
import torch.nn as nn
import visdom
from vae_plots import plot_latent_with_class, save_loss, plot_latent_heatmap_with_class
import torchvision
from torchvision import datasets, transforms
import pyro
import matplotlib.pyplot as plt

from ae import ConvAE
from data_loader import PCA_IM_dataset

from utils.losses import AeLoss


def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 10
    z_dim = 100

    trainset = PCA_IM_dataset(path="./Methods/PCA/results/pca_images/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv", input_dimension=568)
    #trainset = DataSet2(path="./data/data_median_all_label.csv")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    eval_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=False, num_workers=2)
    # setup the VAE
    model = ConvAE(image_channels=1, z_dim=z_dim)
    if args.cuda:
        model.cuda()

    for p in model.parameters():
        print(p.name, p.data.size(), p.requires_grad)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_recon_elbo = []
    eval_recon_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_recon_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            #print(x.size(1))
            if x.size(0) != batch_size:
                continue
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            # x = x.reshape(-1, 1, 28, 28)
            # x_im = x[0, 0, :, :].detach().cpu().numpy()
            # cv2.imshow("im", x_im)
            # cv2.waitKey(0)
            optimizer.zero_grad()
            m, recon = model(x)
            # epoch_loss += svi.step(x)
            loss = mse_loss = AeLoss(recon, x)
            epoch_recon_loss += mse_loss
            loss.backward()
            optimizer.step()

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        train_recon_elbo.append(epoch_recon_loss / normalizer_train)
        print(
            "[epoch %03d]  average training recon loss: %f"
            % (epoch, epoch_recon_loss / normalizer_train)
        )

        if epoch == args.tsne_iter:
            torch.save(model.state_dict(), args.main_path + 'ae' + str(args.tsne_iter) + '.pth')
            save_loss(np.array(train_recon_elbo), None,
                      np.array(eval_recon_elbo), None, save_path=args.main_path)

            plot_latent_heatmap_with_class(vae=model, test_loader=eval_loader, batch_size=batch_size, z_dim=z_dim, args=args, save_path=args.main_path)

    return model


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-data_path",
        default="",
        type=str,
        help="the path of the data",
    )
    parser.add_argument(
        "-tf",
        "--eval-frequency",
        default=100,
        type=int,
        help="how often we evaluate the eval set",
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
        default=100,
        type=int,
        help="epoch when tsne visualization runs",
    )
    parser.add_argument(
        "--main_path",
        default="./results/after_pca/no_class/ae/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)