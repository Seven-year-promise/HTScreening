import numpy as np

import sys
sys.path.append("../../")
import csv
import os
from sklearn.preprocessing import Normalizer

CLASSES = {"Wildtype": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

RAW_CLASSES = {"Wildtype": 0,
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

def load_action_mode(path):
    """
    return: key (compound) and value (action mode) are both integar
    """
    action_dict = {}
    with open(path, "r") as l_f:
        read_label_lines = csv.reader(l_f, delimiter=",")
        for j, l in enumerate(read_label_lines):
            if j > 0:
                #print(l)
                compound_name = l[0]
                if compound_name[0] == "s":
                    compound_name = "C0"
                action_name = l[-2]

                if compound_name in action_dict:
                    continue
                else:
                    action_dict[int(compound_name[1:])] = int(l[-1])
    return action_dict

def load_cleaned_data(path, label_path, normalize):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)

    with open(label_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_labels = [int(i) for i in l]

    data_list = np.array(data_list)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)

    return data_list, np.array(data_labels)

def load_data(base_path, dim_begin = 0, dim_end = 188):
    WT_path = base_path + "WT/"
    data_dict = {}
    all_data = []
    # read WT data
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
                data_item = list(wt_d_l[dim_begin:dim_end, i])
                data_dict["WT"].append(data_item)
                all_data.append(data_item)

    print(len(data_dict["WT"]))
    data_files = os.listdir(base_path)
    for d_f in data_files:
        if d_f == "WT":
            continue
        d_f_name = d_f[:-4]
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
                data_item = list(d_l[dim_begin:dim_end, i])
                data_dict[d_f_name].append(data_item)
                all_data.append(data_item)

    return data_dict, all_data

def load_train_test_data(data_path, label_path):

    data_list = []
    data_labels = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)

    with open(label_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_labels = [int(i) for i in l]

    x = np.array(data_list)
    y = np.array(data_labels)#.reshape(-1, 1)
    return x, y

def load_cleaned_train_test_data(data_path):

    data_list = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)


    x = np.array(data_list)
    return x

def load_feature_data_together(path):
    """
    :param path: the path of data
    :return: the compound names, data, action infos (name and index) in three lists
    """
    comp_names = []
    all_data = []
    action_infos = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l[1:-2]]
            comp_names.append(l[0])
            all_data.append(data_line)
            action_infos.append(l[-2:])

    return comp_names, all_data, action_infos