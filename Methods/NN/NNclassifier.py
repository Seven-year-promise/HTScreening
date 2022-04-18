import argparse
import numpy as np
import torch
import torch.nn as nn
import pyro
import os
from nn import NN
from data_loader import EffectedDataSet
from data_loader import CLASSES
import pandas as pd

criterion = nn.BCELoss(reduction='sum')
MSE = nn.MSELoss(reduction="sum")
cross_entropy = nn.CrossEntropyLoss(reduction="sum")
def VaeLoss(recon_x,x,mu,logvar):
    MSE_loss = MSE(recon_x, x)#/10000000
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# * 100.
    loss = MSE_loss+KLD
    return loss, MSE_loss, KLD

def classifier_loss(output, label):
    return cross_entropy(output, label)

def main(args):
    # clear param store
    pyro.clear_param_store()

    batch_size = 100
    z_dim = 30
    print("load training ...")
    trainset = EffectedDataSet(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_dataset/train_set.csv", normalize=False)
    print("load evaling ...")
    evalset = EffectedDataSet(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_dataset/eval_set.csv", normalize=False)
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
    model = NN(input_dim=541, z_dim=z_dim, classes=len(CLASSES)-1)
    if args.cuda:
        model.cuda()

    for p in model.parameters():
        print(p.name, p.data.size(), p.requires_grad)

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sm = nn.Softmax(dim=1)
    train_loss = []
    train_acc = []
    eval_loss = []
    eval_acc = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        ave_loss = 0.0
        correct_pred = 0
        total_pred = 0
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
            classification = model(x)
            loss = classifier_loss(classification, l)
            ave_loss += loss
            loss.backward()
            optimizer.step()

            outputs = sm(classification)
            probs, predictions = torch.max(outputs, 1)
            for label, prediction in zip(l, predictions):
                if label == prediction:
                    correct_pred += 1
                total_pred += 1
        ave_acc = (correct_pred * 100.0) / total_pred

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        train_loss.append(ave_loss / normalizer_train)
        train_acc.append(ave_acc)
        print(
            "[epoch %03d]  average training loss: %f"
            % (epoch, ave_loss / normalizer_train)
        )
        print(
            "\b [epoch %03d]  -----------average traing acc: %f"
            % (epoch, ave_acc)
        )


        if epoch % args.eval_frequency == 0:
            t_ave_loss = 0.0
            correct_pred = 0
            total_pred = 0
            y_actual = []
            y_pred = []
            for t_x, t_l in eval_loader:
                # if on GPU put mini-batch into CUDA memory
                if t_x.size(0) != batch_size:
                    continue
                if args.cuda:
                    t_x = t_x.cuda()
                    t_l = t_l.cuda()
                # do ELBO gradient and accumulate loss
                # x = x.reshape(-1, 1, 28, 28)
                # x_im = x[0, 0, :, :].detach().cpu().numpy()
                # cv2.imshow("im", x_im)
                # cv2.waitKey(0)
                #optimizer.zero_grad()
                classification = model(t_x)
                t_loss = classifier_loss(classification, t_l)
                #t_class_loss *= 1000
                t_ave_loss += t_loss
                #loss.backward()
                #optimizer.step()
                outputs = sm(classification)
                probs, predictions = torch.max(outputs, 1)
                for label, prediction in zip(t_l, predictions):
                    #print(label.cpu().numpy(), prediction.cpu().numpy())
                    y_actual.append(label.cpu().numpy())
                    y_pred.append(prediction.cpu().numpy())
                    if label == prediction:
                        correct_pred += 1
                    total_pred += 1
            t_ave_acc = (correct_pred * 100.0) / total_pred
            y_actu = pd.Series(np.array(y_actual).reshape(-1, ), name='Actual')
            y_pred = pd.Series(np.array(y_pred).reshape(-1, ), name='Predicted')
            df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
            print(df_confusion)
            normalizer_eval = len(eval_loader.dataset)
            eval_loss.append(t_ave_loss / normalizer_eval)
            eval_acc.append(t_ave_acc)
            print(
                "[epoch %03d] ---------- average eval loss: %f"
                % (epoch, t_ave_loss / normalizer_eval)
            )
            print(
                "\b [epoch %03d]  -----------average eval acc: %f"
                % (epoch, t_ave_acc)
            )

        if epoch == args.tsne_iter:
            torch.save(model.state_dict(), args.main_path + 'models/nn' + str(args.tsne_iter) + '.pth')
            with open(os.path.join(args.main_path + "models/", "train_loss.txt"), "a+") as f:
                for i, (t_l, t_a) in enumerate(zip(train_loss, train_acc)):
                    f.write("Epoch {}    train loss: {}   train acc: {}  \n".format(i, t_l, t_a))

            with open(os.path.join(args.main_path + "models/", "eval_loss.txt"), "a+") as f:
                for i, (e_l, e_a) in enumerate(zip(eval_loss, eval_acc)):
                    f.write("Epoch {}    eval loss: {}   eval acc: {} \n".format(i, e_l, e_a))

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
        "--eval_frequency",
        default=1,
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
        default="./results/split/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)