import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer

LATENT_DIMENSION = 2

CLASSES = {"WT_control": 0,
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

def load_data():
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train = (x_train.astype(np.float64) / 255.0) - 0.5
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_eval = (x_eval.astype(np.float64) / 255.0) - 0.5
    x_eval = x_eval.reshape(x_eval.shape[0], -1)
    y_train = y_train.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)
    return x_train, y_train, x_eval, y_eval

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

