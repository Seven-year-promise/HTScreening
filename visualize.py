import os
import math
import csv
import sys
from skimage import io, draw, color, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from Methods.PCA.utils import load_data
from Methods.PCA.pca_dim_reduce import try_PCA_with_torch
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

def visualize_compound_cleaned(compounds=[], input_dimension = 568):
    base_path = "/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/"
    WT_path = base_path + "WT/"
    data_dict = {}
    #read WT data
    WT_files = os.listdir(WT_path)
    data_dict["WT"] = []
    for wt_f in WT_files:
        # print(d_f_name, lb)
        wt_d_l = []
        print("reading: ", WT_path + wt_f)
        with open(WT_path + wt_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter=",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    wt_d_l.append(data_line)
            wt_d_l = np.array(wt_d_l, np.float)
            for i in range(wt_d_l.shape[1]):
                # print(list(d_l[:, i]))
                data_dict["WT"].append(list(wt_d_l[:input_dimension, i]))

    data_files = os.listdir(base_path)
    for d_f in data_files:
        if d_f == "WT":
            continue
        d_f_name = d_f[:-4]
        if d_f_name[:2] == "WT":
            d_f_name = "WT"
        # print(d_f_name, lb)
        data_dict[d_f_name] = []
        d_l = []
        print("reading: ", base_path + d_f)
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
        for d in data_dict[compounds[i]][::8]:
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


if __name__ == "__main__":
    #visualize(path="./data/raw_data/old_compounds/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv")
    #visualize_PCA()
    visualize_compound_cleaned(["WT", "C5", "C105"], input_dimension=541) #, "C14", "C20", "C88"])
    #visualize_compound_after_PCA(["WT", "C5", "C12", "C88", "C105", "C117"])