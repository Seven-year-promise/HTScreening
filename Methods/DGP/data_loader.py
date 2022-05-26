import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer


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
    all_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):


            one_data = [float(i) for i in l[1:-2]]
            if len(one_data) != 539:
                print("oops!")
                continue
            all_comps.append(l[0])
            #print(len(one_data))
            all_data.append(one_data)
            all_labels.append(int(l[-1]))
    #print(all_data)
    all_data = np.array(all_data)
    all_comps = np.array(all_comps)
    all_labels = np.array(all_labels)
    print(all_data.shape)
    return all_data, all_comps, all_labels


# ---------------before---------------------------------
def combine_data(data1, data2, label1, label2):
    num1 = data1.shape[0]
    num2 = data2.shape[0]

    all_data = np.zeros((num1+num2, data1.shape[1]))
    all_labels = np.zeros((num1+num2))

    all_data[:num1, :] = data1
    all_labels[:num1] = label1
    all_data[num1:, :] = data2
    all_labels[num1:] = label2

    return all_data, all_labels

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

def load_data():
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train = (x_train.astype(np.float64) / 255.0) - 0.5
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_eval = (x_eval.astype(np.float64) / 255.0) - 0.5
    x_eval = x_eval.reshape(x_eval.shape[0], -1)
    y_train = y_train.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)
    return x_train, y_train, x_eval, y_eval

"""
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
"""

def load_effected_data_comp_action(path, normalize):
    data_list = []
    data_labels = []
    actions = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                comp_name = l[0].split("_")[0]
                if comp_name[0] == "W":
                    comp = 0
                else:
                    comp = int(comp_name[1:])
                data_line = [float(i) for i in l[1:-1]]
                data_list.append(data_line)
                data_labels.append(comp)
                actions.append(CLASSES[l[-1]])

    data_list = np.array(data_list)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)
    print("number of data", len(actions))
    print("dimension of data", data_list.shape[1])
    return data_list, np.array(data_labels).reshape(-1, 1), np.array(actions).reshape(-1, 1)

def load_effected_data(path, normalize):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                data_line = [float(i) for i in l[1:-1]]
                data_list.append(data_line)
                data_labels.append(CLASSES[l[-1]])

    data_list = np.array(data_list)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)
    print("number of data", len(data_labels))
    print("dimension of data", data_list.shape[1])
    return data_list, np.array(data_labels).reshape(-1, 1)

def load_effected_action_data(path, normalize, actions=[]):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l[1:-2]]
            data_list.append(data_line)
            data_labels.append(int(l[-1]))

    data_list = np.array(data_list)
    data_labels = np.array(data_labels)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)
    inds = np.zeros((data_labels.shape[0]), np.bool)
    for a in actions:
        inds = np.logical_or(data_labels == a, inds)

    data_list = data_list[inds, :]
    data_labels = (data_labels[inds]>0) * 1
    #data_labels = data_labels[inds]
    print("binary or not?", np.max(data_labels))
    print("number of data", data_list.shape[0])
    print("dimension of data", data_list.shape[1])
    return data_list, data_labels.reshape(-1, 1)

def load_effected_action_data_dimension(path, normalize, actions=[], del_d=1):
    """
    delete a certain dimension of the data
    """
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l[1:-1]]
            data_list.append(data_line)
            data_labels.append(CLASSES[l[-1]])

    data_list = np.array(data_list)
    data_labels = np.array(data_labels)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)
    inds = np.zeros((data_labels.shape[0]), np.bool)
    for a in actions:
        inds = np.logical_or(data_labels == a, inds)


    data_list = data_list[inds, :]
    data_labels = (data_labels[inds]>0) * 1

    dimensions = []
    for i in range(data_list.shape[1]):
        if i == del_d:
            dimensions.append(False)
        else:
            dimensions.append(True)
    data_list = data_list[:, dimensions]
    print("binary or not?", np.max(data_labels))
    print("number of data", data_list.shape[0])
    print("dimension of data", data_list.shape[1])
    return data_list, data_labels.reshape(-1, 1)

def load_filtered_effected_data(path, normalize):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                data_line = [float(i) for i in l[1:-1]]
                data_list.append(data_line)
                data_labels.append(int(l[-1]))

    data_list = np.array(data_list)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)
    print("number of data", len(data_labels))
    print("dimension of data", data_list.shape[1])
    return data_list, np.array(data_labels).reshape(-1, 1)

