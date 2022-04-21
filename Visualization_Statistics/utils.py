import matplotlib.pyplot as plt
import csv
import numpy as np

CLASSES = {"Wildtype": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

CLASSES_alias = {"WT_control": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

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

def load_feature_data_all_by_compound(path):
    """
    :param path: the path of data
    :return: the dict format of the feature for each compound
    """
    all_data_dict = {}
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            compound_name = l[0]
            if compound_name not in all_data_dict:
                all_data_dict[compound_name] = []
            data_line = [float(i) for i in l[1:-2]]
            all_data_dict[compound_name].append(data_line)

    return all_data_dict


def load_feature_data_all_by_action(path):
    """
    :param path: the path of data
    :return: the dict format of the feature for each compound
    """
    all_data_dict = {}
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            action_name = l[-2]
            if action_name not in all_data_dict:
                all_data_dict[action_name] = []
            data_line = [float(i) for i in l[1:-2]]
            all_data_dict[action_name].append(data_line)

    return all_data_dict

def load_effected_data_feature_median(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                action_name = l[-2]
                if exist_key(CLASSES, action_name):
                    comp_name = l[0]
                    #print(comp_name)
                    if comp_name[0] == "s":
                        compound = 0
                    else:
                       compound = int(comp_name[1:])
                    compounds.append(compound)
                    data_line = [float(i) for i in l[1:-2]]
                    data_list.append(data_line)
                    data_labels.append(CLASSES[l[-2]])

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)

def load_effected_data_feature_median(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                action_name = l[-2]
                if exist_key(CLASSES, action_name):
                    comp_name = l[0]
                    #print(comp_name)
                    if comp_name[0] == "s":
                        compound = 0
                    else:
                       compound = int(comp_name[1:])
                    compounds.append(compound)
                    data_line = [float(i) for i in l[1:-2]]
                    data_list.append(data_line)
                    data_labels.append(CLASSES[l[-2]])

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)

def exist_key(dict, key):
    for k, v in dict.items():
        if k == key:
            return True

    return False

def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return "None"

def load_effected_data_ori(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                comp_name = l[0].split("_")[0]
                #print(comp_name)
                if comp_name[0] == "W":
                    compound = 0
                else:
                   compound = int(comp_name[1:])
                compounds.append(compound)
                data_line = [float(i) for i in l[1:-1]]
                data_list.append(data_line)
                data_labels.append(CLASSES_alias[l[-1]])

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)

def load_effected_data_feature(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                action_name = l[-1]
                if exist_key(CLASSES_alias, action_name):
                    comp_name = l[0].split("_")[0]
                    #print(comp_name)
                    if comp_name[0] == "W":
                        compound = 0
                    else:
                       compound = int(comp_name[1:])
                    compounds.append(compound)
                    data_line = [float(i) for i in l[1:-1]]
                    data_list.append(data_line)
                    data_labels.append(CLASSES_alias[l[-1]])

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)