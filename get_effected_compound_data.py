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


def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return "None"

def load_effected_compounds(path):
    effected_c = {}
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                print(l)
                effected_c[l[0]] = []
                for c_i, c in enumerate(l):
                    if c_i > 1:
                        if c != "":
                            effected_c[l[0]].append(c)
                #data_line = [i for i in l]
                #print(data_line)
                #data_list.append(data_line)
                #data_labels.append(CLASSES[l[-1]])
    print(effected_c)

    return effected_c

def get_effected_compound_data(effected_comp, data_path, save_path):
    all_data = []
    for k in effected_comp.keys():
        comp_name_list = effected_comp[k]
        action_label = CLASSES[k]
        print("for action: ", k, action_label)
        for com_name in comp_name_list:

            print("   for compound: ", com_name)
            if com_name == "C0":
                wt_files = os.listdir(data_path+"WT/")
                for w_f in wt_files:
                    with open(data_path+"WT/"+w_f, newline='') as csv_f:
                        read_lines = csv.reader(csv_f, delimiter=",")
                        wt_d_l = []
                        for j, l in enumerate(read_lines):
                            if j > 0:
                                data_line = [float(i) for i in l]
                                wt_d_l.append(data_line)
                        wt_d_l = np.array(wt_d_l, np.float)
                        print(wt_d_l.shape)

                        for i in range(wt_d_l.shape[1]):
                            comp_data = []
                            comp_data.append(com_name)
                            # print(list(d_l[:, i]))
                            comp_data += list(wt_d_l[:, i])
                            comp_data += [k, action_label]
                            all_data.append(comp_data)
            else:
                if os.path.exists(data_path+com_name+".csv"):
                    with open(data_path+com_name+".csv", newline='') as csv_f:
                        read_lines = csv.reader(csv_f, delimiter=",")
                        d_l = []
                        for j, l in enumerate(read_lines):
                            if j > 0:
                                data_line = [float(i) for i in l]
                                d_l.append(data_line)
                        d_l = np.array(d_l, np.float)
                        print(d_l.shape)
                        for i in range(d_l.shape[1]):
                            comp_data = []
                            comp_data.append(com_name)
                            # print(list(d_l[:, i]))
                            comp_data += list(d_l[:, i])
                            comp_data += [k, action_label]
                            all_data.append(comp_data)
                else:
                    print("this compound does not exist!!!")


    with open(save_path, "w") as all_csv:
        all_csv_writer = csv.writer(all_csv)
        all_csv_writer.writerows(all_data)


if __name__ == "__main__":
    e_comp = load_effected_compounds(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_list_4actions.csv")
    get_effected_compound_data(e_comp, data_path="./data/cleaned_data/", save_path="./data/effected_compounds_cleaned_ori_data.csv")
    #split_data(datas, labels, save_path="./data/effected_dataset/")
