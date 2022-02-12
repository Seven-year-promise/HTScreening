import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import random
import numpy as np
import os
import math
import csv
import sys
from skimage import io, draw, color, transform
import numpy as np
import matplotlib.pyplot as plt

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


def get_data(data_path, input_dimension=568):
    base_path = data_path
    WT_path = base_path + "WT/"
    data_list = []
    data_labels = []
    # read WT data
    WT_files = os.listdir(WT_path)
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
                data_list.append(list(wt_d_l[:input_dimension, i]))
                data_labels.append(["WT"])

    data_files = os.listdir(base_path)
    for d_f in data_files:
        if d_f == "WT":
            continue
        d_f_name = d_f[:-4]
        if d_f_name[:2] == "WT":
            d_f_name = "WT"
        # print(d_f_name, lb)
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
                data_list.append(list(d_l[:input_dimension, i]))
                data_labels.append([d_f_name])

    print("the number of data:", len(data_labels))
    print("the dimension of data:", len(data_list[0]))

    return data_labels, data_list

def split_data(datas, labels, save_path):
    train_set = []
    eval_set = []
    label_train_set = []
    label_eval_set = []
    num = len(datas)
    for r_i in range(num):
        if r_i%3==0 or r_i%5==0 or r_i%7==0 or r_i%9==0:
            eval_set.append(datas[r_i])
            label_eval_set.append(labels[r_i][0])
        else:
            train_set.append(datas[r_i])
            label_train_set.append(labels[r_i][0])

    #train_set = datas[random_inds]
    #print(train_set)

    #eval_set = datas[random_inds[train_num:]]

    #label_train_set = labels[random_inds[:train_num]]
    #label_eval_set = labels[random_inds[train_num:]]
    print("number of training: ", len(train_set), len(label_train_set))
    print("number of evaling: ", len(eval_set), len(label_eval_set))

    # write for datas
    with open(save_path + "all.csv", "w") as all_csv:
        all_csv_writer = csv.writer(all_csv)
        all_csv_writer.writerows(datas)

    with open(save_path + "train_set.csv", "w") as train_csv:
        train_csv_writer = csv.writer(train_csv)
        train_csv_writer.writerows(train_set)

    with open(save_path + "eval_set.csv", "w") as eval_csv:
        eval_csv_writer = csv.writer(eval_csv)
        eval_csv_writer.writerows(eval_set)

    # write for labels
    with open(save_path + "all_label.csv", "w") as all_label_csv:
        all_label_csv_writer = csv.writer(all_label_csv)
        all_label_csv_writer.writerows(labels)

    with open(save_path + "train_label.csv", "w") as train_label_csv:
        train_label_csv_writer = csv.writer(train_label_csv)
        train_label_csv_writer.writerow(label_train_set)

    with open(save_path + "eval_label.csv", "w") as eval_label_csv:
        eval_label_csv_writer = csv.writer(eval_label_csv)
        eval_label_csv_writer.writerow(label_eval_set)

if __name__ == "__main__":
    labels, datas = get_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/",
                         input_dimension=541)

    split_data(datas, labels, save_path="./data/cleaned_dataset/")
