import sys
#sys.path.insert(0, "/srv/yanke/PycharmProjects/HTScreening")
import numpy as np

import sys
sys.path.append("../../")
import csv
from Methods.PCA.PCA import PCA_torch
from Methods.PCA.plot_tsne import plot_dist_no_label, plot_dist_with_label, plot_dist_name, plot_dist_train_test, plot_tsne_train_test
import os
import matplotlib.pyplot as plt
from Methods.PCA.utils import load_data


def try_PCA_with_test(train_set, test_set):
    pca = PCA_torch(center=False, n_components=2)
    new_train = pca.fit_PCA(train_set)
    new_test = pca.test(test_set)

    return new_train, new_test

def try_pca_on_compounds(compounds=[], data_dict={}):
    PCA_dim = 1
    pca = PCA_torch(center=False, n_components=PCA_dim)



    #dis_num = 0
    #for c in compounds:
    #    dis_num += len(data_dict[c])

    display_data = []#np.zeros((dis_num, 10), np.float)
    #fig, axs = plt.subplots(len(compounds)+1)
    fig = plt.figure()
    for i in range(len(compounds)):

        comp_data = data_dict[compounds[i]]
        num_data = len(comp_data)
        print("number of data for compound " + compounds[i], num_data)
        comp_data = np.array(comp_data).T
        new_comp_data = pca.fit_PCA(comp_data)
        new_comp_data = np.squeeze(new_comp_data.reshape(-1, 1), axis=1)
        print(new_comp_data)
        display_data.append(new_comp_data)
        #axs[i].imshow(new_comp_data, cmap="hot", interpolation="nearest")
        #axs[i].set_ylabel("fish case")
        #axs[i].set_xlabel("dimension")
        #axs[i].set_title(compounds[i])
        #plt.plot(plot_data[i, :])
    #plt.xlabel("time")
    #plt.title("motion index")
    #axs[-1].imshow(np.array(display_data), cmap="hot", interpolation="nearest")
    #axs[-1].set_ylabel("fish case")
    #axs[-1].set_xlabel("dimension")
    #axs[-1].set_title("all")
    for d, c in zip(display_data, compounds):
        plt.plot(d, label=c)
    #plt.colorbar(im)
    plt.ylabel("Fish case")
    plt.xlabel("Components of PCA")
    plt.title("all")
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()


if __name__ == '__main__':
    #x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
    #                             label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    #x_eval, y_eval = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_set.csv",
    #                           label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_label.csv")

    data_dict, _ = load_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/", dim_begin = 0, dim_end = 541)
    try_pca_on_compounds(["WT", "C5", "C6", "C10", "C11", "C45", "C46", "C55", "C56", "C85", "C95", "C105", "C106", "C111", "C112", "C117"], data_dict)