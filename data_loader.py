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
from sklearn.preprocessing import Normalizer
import skimage.io as io

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

class EffectedDataSetSplited(data.Dataset):
    def __init__(self, path, label_path, normalize=False):
        # read the label file, total: 4 classes
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
        self.data = data_list.tolist()

        self.labels = data_labels
        print("the number of data:", len(data_labels))
        print("the dimension of data:", len(data_list[0]))

    def __getitem__(self, index):

        d = torch.from_numpy(np.array(self.data[index])).float()
        l = torch.from_numpy(np.array(self.labels[index])).long()
        return d, l

    def __len__(self):
        return len(self.data)

class EffectedDataSet(data.Dataset):
    def __init__(self, path, label_path, input_dimension=568):
        # read the label file, total: 10 classes
        label_dict = {}
        with open(label_path, "r") as l_f:
            read_label_lines = csv.reader(l_f, delimiter = ",")
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
            #print(d_f_name, lb)
            d_l = []
            with open(path + d_f, newline='') as csv_f:
                read_lines = csv.reader(csv_f, delimiter = ",")
                for j, l in enumerate(read_lines):
                    if j > 1:
                        data_line = [float(i) for i in l]
                        d_l.append(data_line)

                d_l = np.array(d_l, np.float)
                for i in range(d_l.shape[1]):
                    #print(list(d_l[:, i]))
                    data_list.append(list(d_l[:input_dimension, i]))
                    data_labels.append(lb)

        #data_list = np.array(data_list)
        #max_v = np.max(data_list, 0)
        #min_v = np.min(data_list, 0)
        #print(max_v, min_v)
        #data_list = (data_list - min_v) / (max_v - min_v)
        self.data = data_list

        self.labels = data_labels
        print("the number of data:", len(data_labels))
        print("the dimension of data:", len(data_list[0]))

    def __getitem__(self, index):

        d = torch.from_numpy(np.array(self.data[index])).float()
        l = torch.from_numpy(np.array(self.labels[index])).long()
        return d, l

    def __len__(self):
        return len(self.data)


class RawDataSet(data.Dataset):
    def __init__(self, path, label_path, input_dimension=568):
        # read the label file, total: 10 classes
        label_dict = {}
        with open(label_path, "r") as l_f:
            read_label_lines = csv.reader(l_f, delimiter = ",")
            for j, l in enumerate(read_label_lines):
                if j > 0:
                    label_dict[l[0]] = int(l[-1])
        data_list = []
        data_labels = []
        data_files = os.listdir(path)
        for d_f in data_files:
            d_f_name = d_f[:-6]
            if d_f_name[:2] == "WT":
                lb = 0
            else:
                lb = label_dict[d_f_name]
            #print(d_f_name, lb)
            d_l = []
            with open(path + d_f, newline='') as csv_f:
                read_lines = csv.reader(csv_f, delimiter = ",")
                for j, l in enumerate(read_lines):
                    if j > 1:
                        data_line = [float(i) for i in l]
                        d_l.append(data_line)

                d_l = np.array(d_l, np.float)
                for i in range(d_l.shape[1]):
                    #print(list(d_l[:, i]))
                    data_list.append(list(d_l[:input_dimension, i]))
                    data_labels.append(lb)

        #data_list = np.array(data_list)
        #max_v = np.max(data_list, 0)
        #min_v = np.min(data_list, 0)
        #print(max_v, min_v)
        #data_list = (data_list - min_v) / (max_v - min_v)
        self.data = data_list

        self.labels = data_labels
        print("the number of data:", len(data_labels))
        print("the dimension of data:", len(data_list[0]))

    def __getitem__(self, index):

        d = torch.from_numpy(np.array(self.data[index])).float()
        l = torch.from_numpy(np.array(self.labels[index])).long()
        return d, l

    def __len__(self):
        return len(self.data)

class DataSet(data.Dataset):
    def __init__(self, path, label_path):
        # read the label file, total: 10 classes
        label_dict = {}
        with open(label_path, "r") as l_f:
            read_label_lines = csv.reader(l_f, delimiter = ",")
            for j, l in enumerate(read_label_lines):
                if j > 0:
                    label_dict[l[0]] = int(l[-1])
        data_list = []
        data_labels = []
        data_files = os.listdir(path)
        for d_f in data_files:
            d_f_name = d_f[:-4]
            if d_f_name[:2] == "WT":
                lb = 0
            else:
                lb = label_dict[d_f_name]
            #print(d_f_name, lb)
            with open(path + d_f, newline='') as csv_f:
                read_lines = csv.reader(csv_f, delimiter = ",")
                for j, l in enumerate(read_lines):
                    if j > 0:
                        data_line = [float(i) for i in l]
                        data_list.append(data_line)
                        data_labels.append(lb)
                        #print(data_line)

        data_list = np.array(data_list)
        max_v = np.max(data_list, 0)
        min_v = np.min(data_list, 0)
        print(max_v, min_v)
        data_list = (data_list - min_v) / (max_v - min_v)
        self.data = data_list.tolist()

        self.labels = data_labels
        print("the number of data:", len(data_labels))

    def __getitem__(self, index):

        d = torch.from_numpy(np.array(self.data[index])).float()
        l = torch.from_numpy(np.array(self.labels[index])).long()
        return d, l

    def __len__(self):
        return len(self.data)


class DataSet2(data.Dataset):
    def __init__(self, path):
        # read the label file, total: 10 classes
        data_list = []
        data_labels = []
        with open(path, "r") as l_f:
            read_label_lines = csv.reader(l_f, delimiter = ",")
            for j, l in enumerate(read_label_lines):
                if j > 0:
                    data_line = [float(i) for i in l[1:-2]]
                    data_list.append(data_line)
                    data_labels.append(int(l[-1]))
        print(data_list, data_labels)

        self.data = data_list
        self.labels = data_labels
        print("the number of data:", len(data_labels))

    def __getitem__(self, index):

        d = torch.from_numpy(np.array(self.data[index])).float()
        l = torch.from_numpy(np.array(self.labels[index])).long()
        return d, l

    def __len__(self):
        return len(self.data)

class PCA_IM_dataset(data.Dataset):
    def __init__(self, path):
        # read the label file, total: 4 classes
        data_path_list = []
        im_files = os.listdir(path)
        for i_f in im_files:
            print(path+i_f)
            data_path_list.append(path+i_f)

        self.data_path = data_path_list
        print("the number of data:", len(data_path_list))

    def __getitem__(self, index):
        #print(self.data_path[index])
        im_data = io.imread(self.data_path[index])/255.0 - 0.5

        d = torch.from_numpy(np.array([im_data])).float()

        return d

    def __len__(self):
        return len(self.data_path)