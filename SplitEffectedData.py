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


def load_effected_data(path):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                data_line = [i for i in l]
                #print(data_line)
                data_list.append(data_line)
                data_labels.append(int(l[-1]))

    return np.array(data_list), np.array(data_labels)

def split_data(datas, labels, save_path):
    d_data = len(datas[0])
    index = ["index"+str(i) for i in range(d_data)]
    train_set = []
    eval_set = []
    train_set.append(index)
    eval_set.append(index)
    print(labels)
    for c in range(len(CLASSES)):
        class_ind = labels==c
        print(class_ind)
        class_data = datas[class_ind, :]
        for t in class_data[0::2, :]:
            train_set.append(t)
        for e in class_data[1::2, :]:
            eval_set.append(e)
    print("number of training: ", len(train_set))
    print("number of evaling: ", len(eval_set))

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

if __name__ == "__main__":
    datas,labels = load_effected_data(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_cleaned_ori_data.csv")

    split_data(datas, labels, save_path="./data/effected_dataset/")
