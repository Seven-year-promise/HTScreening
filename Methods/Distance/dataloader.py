import csv
import numpy as np


RAW_CLASSES = {"Wild type": 0,
               "GABAA pore blocker": 1,
               "vesicular ACh transport antagonist": 2,
               "nAChR orthosteric agonist": 3,
               "nAChR orthosteric antagonist": 4,
               "TRPV agonist": 5,
               "GABAA allosteric antagonist": 6,
               "RyR agonist": 7,
               "Na channel": 8,
               "complex II inhibitor": 9,
               "nAChR allosteric agonist": 10,
               "unknown-likely neurotoxin": 11
               }


def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return "None"

def load_cleaned_data(path):
    all_data = []
    all_comps = []
    all_comp_nums = []
    all_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):

            one_data = [float(i) for i in l[1:-2]]
            if len(one_data) != 539:
                print("oops!")
                continue
            all_comp_nums.append(int(l[0][1:]))
            all_comps.append(l[0])
            #print(len(one_data))
            all_data.append(one_data)
            all_labels.append(int(l[-1]))
    #print(all_data)
    all_data = np.array(all_data)
    all_comps = np.array(all_comps)
    all_labels = np.array(all_labels)
    print(all_data.shape)
    return all_data, all_comps, np.array(all_comp_nums), all_labels

def load_featured_data(path):
    all_data = []
    all_comps = []
    all_comp_nums = []
    all_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):

            one_data = [float(i) for i in l[1:-2]]
            if len(one_data) != 10:
                print("oops!")
                continue
            all_comp_nums.append(int(l[0][1:]))
            all_comps.append(l[0])
            #print(len(one_data))
            all_data.append(one_data)
            all_labels.append(int(l[-1]))
    #print(all_data)
    all_data = np.array(all_data)
    all_comps = np.array(all_comps)
    all_labels = np.array(all_labels)
    print(all_data.shape)
    return all_data, all_comps, np.array(all_comp_nums), all_labels