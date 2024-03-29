import argparse
import numpy as np
import torch
import torch.nn as nn
from vae_plots import plot_distribution, save_loss
import torchvision
import pyro
import matplotlib.pyplot as plt

from Methods.AE.ae import AE_class
from Methods.VAE.data_loader import DataSet, RawDataSet, EffectedDataSet, EffectedDataSetSplited
from Methods.VAE.data_loader import CLASSES as CLASSES


criterion = nn.BCELoss(reduction='sum')
MSE = nn.MSELoss(reduction="sum")
cross_entropy = nn.CrossEntropyLoss(reduction="sum")
def AeLoss(recon_x,x):
    MSE_loss = MSE(recon_x, x)#/10000000
    loss = MSE_loss
    return loss

def classifier_loss(output, label):
    return cross_entropy(output, label)

def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 100
    z_dim = 100
    print("load training ...")
    trainset = EffectedDataSetSplited(path="./data/dataset/eval_set.csv", label_path="./data/dataset/eval_label.csv", normalize=True)
    print("load evaling ...")
    evalset = EffectedDataSetSplited(path="./data/dataset/train_set.csv", label_path="./data/dataset/train_label.csv", normalize=True)
    #trainset = EffectedDataSet(path="./data/raw_data/old_compounds/", label_path="./data/raw_data/effected_compounds_pvalue_frames_labeled.csv",
    #                      input_dimension=568)
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv",
    #                      input_dimension=568)
    # trainset = DataSet2(path="./data/data_median_all_label.csv")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    # setup the VAE
    model = AE_class(input_dim=568, h_dim=500, z_dim=z_dim, classes=len(CLASSES))
    if args.cuda:
        model.cuda()

    for p in model.parameters():
        print(p.name, p.data.size(), p.requires_grad)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_recon_elbo = []
    train_class_elbo = []
    eval_recon_elbo = []
    eval_class_elbo = []
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
            recon, classification = model(x)
            # epoch_loss += svi.step(x)
            ae_loss = AeLoss(recon, x)
            class_loss = classifier_loss(classification, l)
            #class_loss *= 1000
            epoch_recon_loss += ae_loss
            epoch_class_loss += class_loss
            loss = ae_loss + class_loss
            loss.backward()
            optimizer.step()

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        train_recon_elbo.append(epoch_recon_loss / normalizer_train)
        train_class_elbo.append(epoch_class_loss / normalizer_train)
        print(
            "[epoch %03d]  average training recon loss: %f  average training class loss: %f"
            % (epoch, epoch_recon_loss / normalizer_train, epoch_class_loss / normalizer_train)
        )

        epoch_t_recon_loss = 0.0
        epoch_t_kld_loss = 0.0
        epoch_t_class_loss = 0.0
        if epoch % args.eval_frequency == 0:
            for t_x, l in eval_loader:
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
                t_recon, t_classification = model(t_x)
                # epoch_loss += svi.step(x)
                t_ae_loss = AeLoss(t_recon, t_x)
                t_class_loss = classifier_loss(t_classification, l)
                #t_class_loss *= 1000
                epoch_t_recon_loss += t_ae_loss
                epoch_t_class_loss += t_class_loss
                #loss.backward()
                #optimizer.step()

            normalizer_eval = len(eval_loader.dataset)
            print(
                "[epoch %03d] ---------- average evaling recon loss: %f  average tesing class loss: %f"
                % (epoch, epoch_t_recon_loss / normalizer_eval, epoch_t_class_loss / normalizer_eval)
            )

        if epoch == args.tsne_iter:
            torch.save(model.state_dict(), args.main_path + 'vae' + str(args.tsne_iter) + '.pth')
            save_loss(np.array(train_recon_elbo), None,
                      np.array(eval_recon_elbo), None, np.array(train_class_elbo), np.array(eval_class_elbo), save_path=args.main_path)
            plot_distribution(vae=model, eval_loader=train_loader, batch_size=batch_size, z_dim=z_dim, args=args,
                              save_path=args.main_path + "train/")
            plot_distribution(vae=model, eval_loader=eval_loader, batch_size=batch_size, z_dim=z_dim, args=args, save_path=args.main_path + "eval/")

    return model


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=401, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-data_path",
        default="",
        type=str,
        help="the path of the data",
    )
    parser.add_argument(
        "-tf",
        "--eval_frequency",
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
        default=400,
        type=int,
        help="epoch when tsne visualization runs",
    )
    parser.add_argument(
        "--main_path",
        default="./results/after_split/with_class/ae_class/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)