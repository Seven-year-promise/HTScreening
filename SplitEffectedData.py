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
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [i for i in l]
            #print(data_line)
            data_list.append(data_line)

    return np.array(data_list)

def split_data(datas, save_path):
    #d_data = len(datas[0])
    #index = ["index"+str(i) for i in range(d_data)]
    train_set = datas[::2, :].tolist()
    eval_set = datas[1::2, :].tolist()
    """
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
    """
    print("number of training: ", len(train_set))
    print("number of evaling: ", len(eval_set))

    # write for datas

    with open(save_path + "effected_feature_train_set.csv", "w") as train_csv:
        train_csv_writer = csv.writer(train_csv)
        train_csv_writer.writerows(train_set)

    with open(save_path + "effected_feature_eval_set.csv", "w") as eval_csv:
        eval_csv_writer = csv.writer(eval_csv)
        eval_csv_writer.writerows(eval_set)

if __name__ == "__main__":
    data = load_effected_data(path="/srv/yanke/PycharmProjects/HTScreening/data/effected/effected_compounds_feature_fish_with_action.csv")

    split_data(data, save_path="/srv/yanke/PycharmProjects/HTScreening/data/effected/")
