import torch
import cv2
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pickle

CLASSES = {"WT_control": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

RAW_CLASSES = {"WT_control": 0,
               "GABAA pore blocker": 1,
               "vesicular ACh transport antagonist": 2,
               "nAChR orthosteric agonist": 3,
               "nAChR orthosteric antagonist": 4,
               "TRPV agonist": 5,
               "GABAA allosteric antagonist": 6,
               "RyR agonist": 7,
               "Na channel": 8,
               "unknown": 9
               }
def plot_dist_train_test(train_feature=None, eval_feature=None, save_path="./vae_results/"):
    fig = plt.figure()
    if train_feature is not None:
        color = plt.cm.Set1(0)
        plt.scatter(train_feature[:, 0], train_feature[:, 1], s=1, color=color, label="train")
    if eval_feature is not None:
        color = plt.cm.Set1(1)
        plt.scatter(eval_feature[:, 0], eval_feature[:, 1], s=1, color=color, label="eval")
    plt.title("First two main dimensions after PCA")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    fig.savefig(save_path +  "VAE_embedding.png")
    plt.clf()

def plot_tsne_train_test(train_feature=None, eval_feature=None, save_path="./vae_results/"):
    fig = plt.figure()
    model_tsne = TSNE(n_components=2, random_state=0)

    if train_feature is not None:
        train_feature = model_tsne.fit_transform(train_feature)
        color = plt.cm.Set1(0)
        plt.scatter(train_feature[:, 0], train_feature[:, 1], s=1, color=color, label="train")
    if eval_feature is not None:
        eval_feature = model_tsne.fit_transform(eval_feature)
        color = plt.cm.Set1(1)
        plt.scatter(eval_feature[:, 0], eval_feature[:, 1], s=1, color=color, label="eval")
    plt.title("First two main dimensions after PCA")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    fig.savefig(save_path +  "VAE_embedding.png")
    plt.clf()

def plot_dist_name(data_x, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    model_tsne = TSNE(n_components=2, random_state=0)
    data_tsne = data_x #model_tsne.fit_transform(data_x)
    fig = plt.figure()
    color = plt.cm.Set1(0)
    for i in range(data_tsne.shape[0]):
        #print(data_tsne[i, 0], data_tsne[i, 1])
        plt.scatter(data_tsne[i, 0], data_tsne[i, 1], s=1, color=color) #, label="all_data")
        plt.text(data_tsne[i, 0], data_tsne[i, 1], i, fontsize=6) #, color=color, label="all_data")
    plt.title("First two main dimensions after PCA")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    #plt.show()
    #fig.savefig(save_path +  "VAE_embedding.png")
    pickle.dump(fig, open(save_path +  "VAE_embedding.pickle", "wb"))
    plt.clf()

def plot_dist_no_label(data_x, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    model_tsne = TSNE(n_components=2, random_state=0)
    data_tsne = data_x #model_tsne.fit_transform(data_x)
    fig = plt.figure()
    color = plt.cm.Set1(0)
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], s=1, color=color, label="all_data")
    plt.title("First two main dimensions after PCA")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    fig.savefig(save_path +  "VAE_embedding.png")
    plt.clf()

def plot_dist_with_label(data_x, data_y, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    model_tsne = TSNE(n_components=2, random_state=0)
    data_tsne = data_x #model_tsne.fit_transform(data_x)
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
        ind_class = data_y == ic
        if np.sum(ind_class*1) > 1:
            color = plt.cm.Set1(ic)
            plt.scatter(data_tsne[ind_class, 0], data_tsne[ind_class, 1], s=10, color=color)
            plt.title("First two main components after PCA (" + action_name + ")")
            fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    #fig.savefig(save_path +  "VAE_embedding.png")
    #fig.clf()
    plt.clf()
    fig = plt.figure()
    for ic in range(len(CLASSES)):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        action_name = get_key(CLASSES, ic)
        ind_class = data_y == ic
        if np.sum(ind_class*1) > 1:
            color = plt.cm.Set1(ic)
            plt.scatter(data_tsne[ind_class, 0], data_tsne[ind_class, 1], s=1, color=color, label=action_name)
    plt.title("First two main dimensions after PCA per Action Mode")
    plt.legend(loc="best")
            #fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    fig.savefig(save_path +  "VAE_embedding.png")
    plt.clf()

def plot_tsne_train_eval_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    color2 = plt.cm.Set1(3)
    for c in range(130):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = z_embed_all["train"][train_inds]
        eval_comp_data = z_embed_all["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue
        fig = plt.figure()

        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.05, label="all")
        plt.scatter(train_comp_data[:, 0], train_comp_data[:, 1], s=5, color=color1, label = "train C" + str(c))
        plt.scatter(np.average(train_comp_data,axis=0)[0], np.average(train_comp_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_comp_data[:, 0], eval_comp_data[:, 1], s=5, color=color2, label="eval C" + str(c))
        plt.scatter(np.average(eval_comp_data, axis=0)[0], np.average(eval_comp_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        plt.legend(loc="best")
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(c) + ".png")
        plt.clf()


def plot_2Ddistribution_train_eval_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    #model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = z_states # model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    color2 = plt.cm.Set1(3)
    for c in range(130):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = z_embed_all["train"][train_inds]
        eval_comp_data = z_embed_all["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue
        fig = plt.figure()

        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.05, label="all")
        plt.scatter(train_comp_data[:, 0], train_comp_data[:, 1], s=5, color=color1, label = "train C" + str(c))
        plt.scatter(np.average(train_comp_data,axis=0)[0], np.average(train_comp_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_comp_data[:, 0], eval_comp_data[:, 1], s=5, color=color2, label="eval C" + str(c))
        plt.scatter(np.average(eval_comp_data, axis=0)[0], np.average(eval_comp_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        #plt.xlim(0.5, 0.9)
        #plt.ylim(-1.5, -0.8)
        plt.legend(loc="best")
        plt.title("Latent Variable first 2D per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(c) + ".png")
        plt.clf()

def plot_2Ddistribution_train_eval_by_action_mode(z_loc, labels, action_modes, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    #model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = z_states # model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    def get_key(dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return "None"
    colors = ["tab:blue", "tab:gray", "tab:pink", "tab:red", "tab:green",
              "tab:purple", "tab:orange", "tab:cyan", "tab:olive", "tab:brown"]

    for a_name, a_num in RAW_CLASSES.items():
        print(a_name, a_num)
        fig = plt.figure()
        color0 = plt.cm.Set1(0)
        color1 = plt.cm.Set1(1)
        color2 = plt.cm.Set1(3)
        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.25, label="all")
        train_action_data = []
        eval_action_data = []
        for c in range(130):
            train_inds = labels["train"] == c
            eval_inds = labels["eval"] == c
            train_comp_data = z_embed_all["train"][train_inds]
            eval_comp_data = z_embed_all["eval"][eval_inds]
            num_train_comp_data = train_comp_data.shape[0]
            num_eval_comp_data = eval_comp_data.shape[0]
            if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
                continue
            if action_modes[c] != a_num:
                continue
            print(c, action_modes[c], a_num)

            for n_t_c in range(num_train_comp_data):
                train_action_data.append(train_comp_data[n_t_c, :])
            for n_e_c in range(num_eval_comp_data):
                eval_action_data.append(eval_comp_data[n_e_c, :])
        train_action_data = np.array(train_action_data)
        eval_action_data = np.array(eval_action_data)

        plt.scatter(train_action_data[:, 0], train_action_data[:, 1], s=5, color=color1, label = a_name)
        plt.scatter(np.average(train_action_data,axis=0)[0], np.average(train_action_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_action_data[:, 0], eval_action_data[:, 1], s=5, color=color2, label=a_name)
        plt.scatter(np.average(eval_action_data, axis=0)[0], np.average(eval_action_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        #plt.xlim(-1.45, -1.3)
        #plt.ylim(-0.7, -0.55)
        plt.legend(loc="best")
        plt.title("Latent Variable first 2D per action mode")
        fig.savefig(save_path + str(name) + "_embedding_action" + str(a_num) + ".png")
        plt.clf()