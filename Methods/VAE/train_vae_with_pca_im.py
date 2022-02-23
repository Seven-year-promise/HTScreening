import argparse
import numpy as np
import torch
import torch.nn as nn
from vae_plots import save_loss, plot_latent_heatmap_with_class
import torchvision
import pyro

from Methods.VAE.vae import ConvVAE
from Methods.VAE.data_loader import PCA_IM_dataset


criterion = nn.BCELoss(reduction='sum')
MSE = nn.MSELoss(reduction="sum")
def VaeLoss(recon_x,x,mu,logvar):
    MSE_loss = MSE(recon_x, x)#/10000000
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# * 100.
    loss = MSE_loss+KLD
    return loss, MSE_loss, KLD


def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 10
    z_dim = 100

    trainset = PCA_IM_dataset(path="../PCA/results/pca_images/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv", input_dimension=568)
    #trainset = DataSet2(path="./data/data_median_all_label.csv")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    eval_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=False, num_workers=2)
    # setup the VAE
    model = ConvVAE(image_channels=1, z_dim=z_dim)
    if args.cuda:
        model.cuda()

    for p in model.parameters():
        print(p.name, p.data.size(), p.requires_grad)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_recon_elbo = []
    train_kld_elbo = []
    eval_recon_elbo = []
    eval_kld_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
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
            m, lvar, recon = model(x)
            # epoch_loss += svi.step(x)
            loss, mse_loss, kld_loss = VaeLoss(recon, x, m, lvar)
            epoch_recon_loss += mse_loss
            epoch_kld_loss += kld_loss
            loss.backward()
            optimizer.step()

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        train_recon_elbo.append(epoch_recon_loss / normalizer_train)
        train_kld_elbo.append(epoch_kld_loss / normalizer_train)
        print(
            "[epoch %03d]  average training recon loss: %f"
            % (epoch, epoch_recon_loss / normalizer_train)
        )
        print(
            " \b[epoch %03d]  average training kld loss: %f"
            % (epoch, epoch_kld_loss / normalizer_train)
        )

        if epoch == args.tsne_iter:
            torch.save(model.state_dict(), args.main_path + 'vae' + str(args.tsne_iter) + '.pth')
            save_loss(np.array(train_recon_elbo), np.array(train_kld_elbo),
                      np.array(eval_recon_elbo), np.array(eval_kld_elbo), save_path=args.main_path)

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
        default="./results/after_pca/no_class/vae/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)