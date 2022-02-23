import torch
import cv2
import numpy as np
from Methods.VAE.data_loader import RAW_CLASSES as CLASSES

def plot_conditional_samples_ssvae(ssvae, visdom_session):
    """
    This is a method to do conditional sampling in visdom
    """
    vis = visdom_session
    ys = {}
    for i in range(10):
        ys[i] = torch.zeros(1, 10)
        ys[i][0, i] = 1
    xs = torch.zeros(1, 784)

    for i in range(10):
        images = []
        for rr in range(100):
            # get the loc from the model
            sample_loc_i = ssvae.model(xs, ys[i])
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)

def save_images(x, recon, epoch, save_path):
    for i, (ori_im, recon_im) in enumerate(zip(x[:, :, :, :], recon[:, :, :, :])):
        ori_im = ori_im.reshape(32, 32).detach().cpu().numpy()
        recon_im = recon_im.reshape(32, 32).detach().cpu().numpy()
        ori_im = np.array(ori_im*255, np.uint8)
        recon_im = np.array(recon_im * 255, np.uint8)
        cv2.imwrite(save_path + "test_ori_images/" + str(epoch) + "_" + str(i) + ".jpg", ori_im)
        cv2.imwrite(save_path + "test_recon_images/" + str(epoch) + "_" + str(i) + ".jpg", recon_im)

def save_loss(recon_loss, kld_loss, recon_test_loss, kld_test_loss, class_loss=None, class_test_loss=None, save_path="./vae_results/"):
    import os

    with open(os.path.join(save_path, "recon_epoch_loss.txt"), "a+") as f:
        for i, t_l in enumerate(recon_loss):
            f.write("Epoch {}    ave train recon loss: {}    \n".format(i, t_l))
        for i, t_l in enumerate(recon_test_loss):
            f.write("Epoch {}    ave test recon loss: {}    \n".format(i, t_l))

    if kld_loss is not None and kld_test_loss is not None:
        with open(os.path.join(save_path, "kld_epoch_loss.txt"), "a+") as f:
            for i, t_l in enumerate(kld_loss):
                f.write("Epoch {}    ave train kld loss: {}    \n".format(i, t_l))
            for i, t_l in enumerate(kld_test_loss):
                f.write("Epoch {}    ave test kld loss: {}    \n".format(i, t_l))

    if class_test_loss is not None and class_loss is not None:
        with open(os.path.join(save_path, "class_epoch_loss.txt"), "a+") as f:
            for i, t_l in enumerate(class_loss):
                f.write("Epoch {}    ave train class loss: {}    \n".format(i, t_l))
            for i, t_l in enumerate(class_test_loss):
                f.write("Epoch {}    ave test class loss: {}    \n".format(i, t_l))

def plot_llk(train_elbo, test_elbo, save_path="./vae_results/"):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    import os

    with open(os.path.join(save_path, "epoch_loss.txt"), "a+") as f:
        for i, t_l in enumerate(train_elbo):
            f.write("Epoch {}    ave train loss: {}    \n".format(i, t_l))
        for i, t_l in enumerate(test_elbo):
            f.write("Epoch {}    ave test loss: {}    \n".format(i, t_l))

    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    data = np.concatenate(
        [np.arange(len(train_elbo))[:, sp.newaxis], -train_elbo[:, sp.newaxis]], axis=1
    )
    df = pd.DataFrame(data=data, columns=["Training Epoch", "Train ELBO"])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Train ELBO")
    g.map(plt.plot, "Training Epoch", "Train ELBO")
    plt.savefig(save_path + "Train_elbo_vae.png")
    plt.close("all")


def plot_vae_samples(vae, visdom_session):
    vis = visdom_session
    x = torch.zeros([1, 784])
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)

def plot_distribution(vae=None, test_loader=None, batch_size=None, z_dim=None, args=None, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    num_data = len(test_loader.dataset)
    print("number of test data: ", num_data)
    z_loc = np.zeros((num_data, z_dim), np.float)
    classes = np.zeros(num_data, np.int)
    for i, (x, c) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
        m, _ = vae.encoder(x)
        if num_data > (i + 1) * batch_size:
            z_loc[(i*batch_size):((i+1)*batch_size), :] = m.detach().cpu().numpy()
            classes[(i*batch_size):((i+1)*batch_size)] = c.detach().cpu().numpy()
        else:
            z_loc[(i*batch_size):, :] = m.detach().cpu().numpy()
            classes[(i*batch_size):] = c.detach().cpu().numpy()


    model_tsne = TSNE(n_components=2, random_state=0)
    z_embed = model_tsne.fit_transform(z_loc)
    def get_key(dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return "None"

    for ic in range(len(CLASSES)):
        fig = plt.figure()
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        action_name = get_key(CLASSES, ic)
        ind_class = classes == ic
        if np.sum(ind_class*1) > 1:
            color = plt.cm.Set1(ic)
            plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
            plt.title("Latent Variable T-SNE (" + action_name + ")")
            fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    #fig.savefig(save_path +  "VAE_embedding.png")
    #fig.clf()
    plt.clf()
    fig = plt.figure()
    for ic in range(len(CLASSES)):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        action_name = get_key(CLASSES, ic)
        ind_class = classes == ic
        if np.sum(ind_class*1) > 1:
            color = plt.cm.Set1(ic)
            plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color, label=action_name)
    plt.title("Latent Variable T-SNE per Action Mode")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    fig.savefig(save_path +  "VAE_embedding.png")
    plt.clf()


def plot_latent_no_class(vae=None, test_loader=None, batch_size=None, z_dim=None, args=None, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    num_data = len(test_loader.dataset)
    print("number of test data: ", num_data)
    z_loc = np.zeros((num_data, z_dim), np.float)
    for i, (x, _) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
        m, _ = vae.encoder(x)
        if num_data > (i + 1) * batch_size:
            z_loc[(i*batch_size):((i+1)*batch_size), :] = m.detach().cpu().numpy()
        else:
            z_loc[(i*batch_size):, :] = m.detach().cpu().numpy()


    model_tsne = TSNE(n_components=2, random_state=0)
    z_embed = model_tsne.fit_transform(z_loc)
    color = plt.cm.Set1(0)
    plt.scatter(z_embed[:, 0], z_embed[:, 1], s=10, color=color)
    plt.title("Latent Variable T-SNE of all data")
    plt.savefig(save_path  + "VAE_embedding_all" + ".png")
    plt.clf()

def plot_latent_with_class(vae=None, test_loader=None, batch_size=None, z_dim=None, args=None, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    num_data = len(test_loader.dataset)
    print("number of test data: ", num_data)
    z_loc = np.zeros((num_data, z_dim), np.float)
    labels = np.zeros((num_data, 1), np.int)
    for i, (x, l) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
        m, _ = vae.encoder(x)
        if num_data > (i + 1) * batch_size:
            z_loc[(i*batch_size):((i+1)*batch_size), :] = m.detach().cpu().numpy()
            labels[(i * batch_size):((i + 1) * batch_size), :] = l
        else:
            z_loc[(i*batch_size):, :] = m.detach().cpu().numpy()
            labels[(i*batch_size):, :] = l

    model_tsne = TSNE(n_components=2, random_state=0)
    z_embed = model_tsne.fit_transform(z_loc)
    color = plt.cm.Set1(0)
    for n in range(num_data):
        plt.scatter(z_embed[n, 0], z_embed[n, 1], s=2, color=color)
        plt.text(z_embed[n, 0], z_embed[n, 1], labels[n, 0], fontsize=5)
    plt.title("Latent Variable T-SNE of all data")
    plt.savefig(save_path  + "VAE_embedding_all" + ".png")
    plt.clf()

def plot_latent_heatmap_with_class(vae=None, test_loader=None, batch_size=None, z_dim=None, args=None, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    num_data = len(test_loader.dataset)
    print("number of test data: ", num_data)
    z_loc = np.zeros((num_data, z_dim), np.float)
    labels = np.zeros((num_data, 1), np.int)
    for i, (x, l) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
        m, _ = vae.encoder(x)
        if num_data > (i + 1) * batch_size:
            z_loc[(i*batch_size):((i+1)*batch_size), :] = m.detach().cpu().numpy()
            labels[(i * batch_size):((i + 1) * batch_size), :] = l
        else:
            z_loc[(i*batch_size):, :] = m.detach().cpu().numpy()
            labels[(i*batch_size):, :] = l

    fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    heatmap = axis.pcolor(z_loc, cmap=plt.cm.Blues)  # heatmap contient les valeurs

    axis.set_yticks(np.arange(z_loc.shape[0]) + 0.5, minor=False)
    axis.set_xticks(np.arange(z_loc.shape[1]) + 0.5, minor=False)

    axis.invert_yaxis()

    axis.set_yticklabels(labels, minor=False)

    fig.set_size_inches(11.03, 3.5)
    plt.title("Latent of all data")
    plt.savefig(save_path  + "VAE_embedding_all" + ".png")
    plt.clf()

def plot_latent_heatmap_by_compound(vae=None, test_loader=None, batch_size=None, z_dim=None, args=None, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    num_data = len(test_loader.dataset)
    print("number of test data: ", num_data)
    z_loc = np.zeros((num_data, z_dim), np.float)
    labels = np.zeros((num_data, 1), np.int)
    for i, (x, l) in enumerate(test_loader):
        if args.cuda:
            x = x.cuda()
        m, _ = vae.encoder(x)
        if num_data > (i + 1) * batch_size:
            z_loc[(i*batch_size):((i+1)*batch_size), :] = m.detach().cpu().numpy()
            labels[(i * batch_size):((i + 1) * batch_size), 0] = l
        else:
            z_loc[(i*batch_size):, :] = m.detach().cpu().numpy()
            labels[(i*batch_size):, 0] = l

    display_data = []
    for c in range(5, 10):
        inds = labels[:, 0] == c
        comp_data = z_loc[inds]
        num_comp_data = comp_data.shape[0]
        if num_comp_data < 1:
            continue
        for c_n in range(num_comp_data):
            display_data.append(comp_data[c_n, :])
        display_data.append(np.zeros(comp_data.shape[1], dtype=np.float))

    fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    #heatmap = axis.pcolor(display_data, cmap=plt.cm.Blues)  # heatmap contient les valeurs
    im = plt.imshow(np.array(display_data), cmap="cool", interpolation="nearest")
    plt.colorbar(im)
    #axis.set_yticks(np.arange(z_loc.shape[0]) + 0.5, minor=False)
    #axis.set_xticks(np.arange(z_loc.shape[1]) + 0.5, minor=False)

    #axis.invert_yaxis()

    #axis.set_yticklabels(labels, minor=False)

    fig.set_size_inches(11.03, 3.5)
    plt.title("Latent of all data")
    plt.savefig(save_path  + "latent_embedding_all" + ".png")
    plt.clf()

def test_tsne(vae=None, test_loader=None, save_path="./vae_results/"):
    """
    This is used to generate a t-sne embedding of the vae
    """
    name = "VAE"
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    data = data.view(-1, 1, 28, 28)#data = data.reshape(-1, 1, 28, 28)
    data = data.cuda()
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, mnist_labels, name, save_path)


def test_tsne_ssvae(name=None, ssvae=None, test_loader=None, save_path="./vae_results/"):
    """
    This is used to generate a t-sne embedding of the ss-vae
    """
    if name is None:
        name = "SS-VAE"
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = ssvae.encoder_z([data, mnist_labels])
    plot_tsne(z_loc, mnist_labels, name, save_path)


def plot_tsne(z_loc, classes, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(ic) + ".png")
    fig.savefig(save_path + str(name) + "_embedding.png")

def visulize(vae, test_loader, save_path):
    """
    This is used to visulize the reconstruction of the images by the method
    """
    data = test_loader.dataset.test_data.float()
    data = data.view(-1, 1, 28, 28)  # data = data.reshape(-1, 1, 28, 28)
    data = data.cuda()
    recon = vae(data)