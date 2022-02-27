import argparse
import os
import numpy as np
import torch
from torchvision import datasets, transforms
import pyro
from nn import NN
from data_loader import EffectedDataSet
from data_loader import CLASSES
from utils import plot_tsne_by_action

z_dim = 30
batch_size = 30

def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup MNIST data loaders
    # train_loader, test_loader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    # validation set and validation data loader
    trainset = EffectedDataSet(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_dataset/train_set.csv", normalize=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    # setup the VAE
    model = NN(input_dim=541, z_dim=z_dim, classes=len(CLASSES))
    if args.cuda:
        model.cuda()
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, "nn100.pth")))
    latent_train = []
    labels = []
    for i, (x, l) in enumerate(train_loader):
        if args.cuda:
            x = x.cuda()
        feature = model.get_latent(x).cpu().detach().numpy()
        l = l.detach().numpy()

        for f_n in range(feature.shape[0]):
            latent_train.append(feature[f_n, :])
            labels.append(l[f_n])

    plot_tsne_by_action(np.array(latent_train), np.array(labels).reshape(-1, 1), name="nn_htscreening_class_train", save_path=args.save_path+"all/")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.7.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-mp", "--model_path", default="./results/split/30-dimension/models/", help="the model path"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="whether to use cuda"
    )
    parser.add_argument(
        "--save_path",
        default="./results/split/30-dimension/",
        help="the path to save",
    )
    args = parser.parse_args()

    model = main(args)