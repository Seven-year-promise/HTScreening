import os
import math
import csv
import sys
from skimage import io, draw, color, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from Methods.PCA.pca_dim_reduce import load_data, try_PCA_with_torch
from Methods.PCA.PCA import PCA_torch

matplotlib.use("TkAgg")

def visualize(path):
    data_lists=[]
    data_files = os.listdir(path)
    for d_f in data_files:
        print(d_f)
        data_list = []
        with open(path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter = ",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    data_list.append(data_line)
        data_lists.append(data_list)

    for d_l in data_lists:
        d_l = np.array(d_l)
        for i in range(d_l.shape[1]):
            plt.plot(d_l[:, i])
        plt.show()

def visualize_PCA():
    x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
                                 label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    x_eval, y_eval = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_set.csv",
                               label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_label.csv")

    x_all = np.ones((x_train.shape[0] + x_eval.shape[0], x_train.shape[1]), dtype=x_train.dtype)
    x_all[:x_train.shape[0], :] = x_train
    x_all[:x_eval.shape[0], :] = x_eval
    _, new_data = try_PCA_with_torch(x_train)

    #print(x_train.shape, np.where(new_data[:, 1]>13)[0])
    plot_data = x_train[np.where(new_data[:, 1]>13.5)[0], :]
    print(plot_data.shape)
    fig, axs = plt.subplots(plot_data.shape[0])
    for i in range(plot_data.shape[0]):
        axs[i].plot(plot_data[i, :])
        #plt.plot(plot_data[i, :])
    plt.xlabel("time")
    plt.ylabel("motion index")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()

def visualize_compound(compounds=[], input_dimension = 568):
    base_path = "/srv/yanke/PycharmProjects/HTScreening/data/raw_data/old_compounds/"

    data_dict = {}
    data_files = os.listdir(base_path)
    for d_f in data_files:
        d_f_name = d_f[:-6]
        if d_f_name[:2] == "WT":
            d_f_name = "WT"
        # print(d_f_name, lb)
        data_dict[d_f_name] = []
        d_l = []
        with open(base_path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter=",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    d_l.append(data_line)
            d_l = np.array(d_l, np.float)
            for i in range(d_l.shape[1]):
                # print(list(d_l[:, i]))
                data_dict[d_f_name].append(list(d_l[:input_dimension, i]))

    fig, axs = plt.subplots(len(compounds))
    for i in range(len(compounds)):
        for d in data_dict[compounds[i]]:
            axs[i].plot(d)
        axs[i].set_ylabel("motion index")
        axs[i].set_xlabel("time")
        axs[i].set_title(compounds[i])
        #plt.plot(plot_data[i, :])
    #plt.xlabel("time")
    #plt.title("motion index")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()

def visualize_compound_after_PCA(compounds=[], input_dimension = 568):
    PCA_dim = 50
    base_path = "/srv/yanke/PycharmProjects/HTScreening/data/raw_data/old_compounds/"
    pca = PCA_torch(center=False, n_components=PCA_dim)


    all_data = []
    data_dict = {}
    data_files = os.listdir(base_path)
    for d_f in data_files:
        d_f_name = d_f[:-6]
        if d_f_name[:2] == "WT":
            d_f_name = "WT"
        # print(d_f_name, lb)
        data_dict[d_f_name] = []
        d_l = []
        with open(base_path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter=",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    d_l.append(data_line)
            d_l = np.array(d_l, np.float)
            for i in range(d_l.shape[1]):
                # print(list(d_l[:, i]))
                data_item = list(d_l[:input_dimension, i])
                data_dict[d_f_name].append(data_item)
                all_data.append(data_item)

    all_data = np.array(all_data)
    new_all_train = pca.fit_PCA(all_data)

    #dis_num = 0
    #for c in compounds:
    #    dis_num += len(data_dict[c])

    display_data = []#np.zeros((dis_num, 10), np.float)
    #fig, axs = plt.subplots(len(compounds)+1)
    fig = plt.figure()
    for i in range(len(compounds)):
        comp_data = data_dict[compounds[i]]
        comp_data = np.array(comp_data)
        new_comp_data = pca.test(comp_data)
        for d in range(new_comp_data.shape[0]):
            display_data.append(new_comp_data[d])
        display_data.append(np.zeros((PCA_dim), dtype=np.float))
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
    im = plt.imshow(np.array(display_data), cmap="gray", interpolation="nearest")
    plt.colorbar(im)
    plt.ylabel("Fish case")
    plt.xlabel("Components of PCA")
    plt.title("all")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()

if __name__ == "__main__":
    #visualize(path="./data/raw_data/old_compounds/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv")
    #visualize_PCA()
    #visualize_compound(["WT", "C5", "C12"]) #, "C14", "C20", "C88"])
    visualize_compound_after_PCA(["WT", "C5", "C12", "C88", "C105", "C117"])