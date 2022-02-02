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