import argparse
import numpy as np
import torch
import torch.nn as nn
from vae_plots import plot_latent_heatmap_by_compound, save_loss
import torchvision
from torchvision import datasets, transforms
import pyro
import matplotlib.pyplot as plt

from Methods.AE.ae import AE
from Methods.VAE.data_loader import DataSet, DataSet2, RawDataSet, EffectedDataSet, EffectedDataSetSplited
from Methods.VAE.data_loader import CLASSES as CLASSES


criterion = nn.BCELoss(reduction='sum')
MSE = nn.MSELoss(reduction="sum")
cross_entropy = nn.CrossEntropyLoss(reduction="sum")
def AeLoss(recon_x,x):
    MSE_loss = MSE(recon_x, x)#/10000000
    loss = MSE_loss
    return loss


def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 100
    z_dim = 100
    print("load training ...")
    trainset = EffectedDataSetSplited(path="../../data/cleaned_dataset/train_set.csv",
                                      label_path="../../data/cleaned_dataset/train_label.csv",
                                      normalize=True)
    evalset = EffectedDataSetSplited(path="../../data/cleaned_dataset/eval_set.csv",
                                     label_path="../../data/cleaned_dataset/eval_label.csv",
                                     normalize=False)
    #trainset = EffectedDataSet(path="./data/raw_data/old_compounds/", label_path="./data/raw_data/effected_compounds_pvalue_frames_labeled.csv",
    #                      input_dimension=568)
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv",
    #                      input_dimension=568)
    # trainset = DataSet2(path="./data/data_median_all_label.csv")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    # setup the VAE
    model = AE(input_dim=540, h_dim=256, z_dim=z_dim)
    if args.cuda:
        model.cuda()

    for p in model.parameters():
        print(p.name, p.data.size(), p.requires_grad)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_recon_elbo = []
    test_recon_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0
        epoch_class_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, l in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if x.size(0) != batch_size:
                continue
            if args.cuda:
                x = x.cuda()
                l = l.cuda()
            # do ELBO gradient and accumulate loss
            # x = x.reshape(-1, 1, 28, 28)
            # x_im = x[0, 0, :, :].detach().cpu().numpy()
            # cv2.imshow("im", x_im)
            # cv2.waitKey(0)
            optimizer.zero_grad()
            recon = model(x)
            # epoch_loss += svi.step(x)
            ae_loss = AeLoss(recon, x)
            #class_loss *= 1000
            epoch_recon_loss += ae_loss
            loss = ae_loss
            loss.backward()
            optimizer.step()

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        train_recon_elbo.append(epoch_recon_loss / normalizer_train)
        print(
            "[epoch %03d]  average training recon loss: %f"
            % (epoch, epoch_recon_loss / normalizer_train)
        )

        epoch_t_recon_loss = 0.0
        epoch_t_kld_loss = 0.0
        epoch_t_class_loss = 0.0
        if epoch % args.test_frequency == 0:
            for t_x, l in test_loader:
                # if on GPU put mini-batch into CUDA memory
                if t_x.size(0) != batch_size:
                    continue
                if args.cuda:
                    t_x = t_x.cuda()
                    l = l.cuda()
                # do ELBO gradient and accumulate loss
                # x = x.reshape(-1, 1, 28, 28)
                # x_im = x[0, 0, :, :].detach().cpu().numpy()
                # cv2.imshow("im", x_im)
                # cv2.waitKey(0)
                #optimizer.zero_grad()
                t_recon = model(t_x)
                # epoch_loss += svi.step(x)
                t_ae_loss = AeLoss(t_recon, t_x)
                #t_class_loss *= 1000
                epoch_t_recon_loss += t_ae_loss
                #loss.backward()
                #optimizer.step()

            normalizer_test = len(test_loader.dataset)
            print(
                "[epoch %03d] ---------- average testing recon loss: %f"
                % (epoch, epoch_t_recon_loss / normalizer_test)
            )

        if epoch == args.tsne_iter:
            torch.save(model.state_dict(), args.main_path + 'vae' + str(args.tsne_iter) + '.pth')
            save_loss(np.array(train_recon_elbo), None,
                      np.array(test_recon_elbo), None, None, None, save_path=args.main_path)
            plot_latent_heatmap_by_compound(vae=model, test_loader=train_loader, batch_size=batch_size, z_dim=z_dim, args=args,
                              save_path=args.main_path)
            #plot_latent_heatmap_by_compound(vae=model, test_loader=test_loader, batch_size=batch_size, z_dim=z_dim, args=args, save_path=args.main_path + "test/")

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
        "--test_frequency",
        default=100,
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
        default=100,
        type=int,
        help="epoch when tsne visualization runs",
    )
    parser.add_argument(
        "--main_path",
        default="./results/cleaned/ae/no_class/train/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)