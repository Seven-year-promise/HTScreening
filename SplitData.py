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


def get_data(path, label_path, input_dimension=568):
    # read the label file, total: 10 classes
    label_dict = {}
    with open(label_path, "r") as l_f:
        read_label_lines = csv.reader(l_f, delimiter=",")
        for j, l in enumerate(read_label_lines):
            compound_name = l[0].split("_")[0]
            action_name = l[-1]
            if j > 0:
                if compound_name in label_dict:
                    continue
                else:
                    label_dict[compound_name] = CLASSES[action_name]
    print("labels: ", label_dict)
    data_list = []
    data_labels = []
    data_files = os.listdir(path)
    for d_f in data_files:
        d_f_name = d_f[:-6]
        if d_f_name[:2] == "WT":
            lb = 0
        else:
            if d_f_name in label_dict:
                lb = label_dict[d_f_name]
            else:
                continue
        # print(d_f_name, lb)
        d_l = []
        with open(path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter=",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    d_l.append(data_line)

            d_l = np.array(d_l, np.float)
            for i in range(d_l.shape[1]):
                # print(list(d_l[:, i]))
                data_list.append(list(d_l[:input_dimension, i]))
                data_labels.append(lb)

    # data_list = np.array(data_list)
    # max_v = np.max(data_list, 0)
    # min_v = np.min(data_list, 0)
    # print(max_v, min_v)
    # data_list = (data_list - min_v) / (max_v - min_v)
    print("the number of data:", len(data_labels))
    print("the dimension of data:", len(data_list[0]))

    return data_labels, data_list

def split_data(datas, labels, save_path):
    train_set = []
    test_set = []
    label_train_set = []
    label_test_set = []
    num = len(datas)
    for r_i in range(num):
        if r_i%9==0 or r_i%10==0:
            test_set.append(datas[r_i])
            label_test_set.append(labels[r_i])
        else:
            train_set.append(datas[r_i])
            label_train_set.append(labels[r_i])

    #train_set = datas[random_inds]
    #print(train_set)

    #test_set = datas[random_inds[train_num:]]

    #label_train_set = labels[random_inds[:train_num]]
    #label_test_set = labels[random_inds[train_num:]]
    print("number of training: ", len(train_set), len(label_train_set))
    print("number of testing: ", len(test_set), len(label_test_set))
    with open(save_path + "train_set.csv", "w") as train_csv:
        train_csv_writer = csv.writer(train_csv)
        train_csv_writer.writerows(train_set)

    with open(save_path + "test_set.csv", "w") as test_csv:
        test_csv_writer = csv.writer(test_csv)
        test_csv_writer.writerows(test_set)

    with open(save_path + "train_label.csv", "w") as train_label_csv:
        train_label_csv_writer = csv.writer(train_label_csv)
        train_label_csv_writer.writerow(label_train_set)

    with open(save_path + "test_label.csv", "w") as test_label_csv:
        test_label_csv_writer = csv.writer(test_label_csv)
        test_label_csv_writer.writerow(label_test_set)

if __name__ == "__main__":
    labels, datas = get_data(path="./data/raw_data/old_compounds/",
                         label_path="./data/raw_data/effected_compounds_pvalue_frames_labeled.csv",
                         input_dimension=568)

    split_data(datas, labels, save_path="./data/dataset/")
