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



class EffectedDataSet(data.Dataset):
    def __init__(self, path, normalize=False):
        # read the label file, total: 10 classes
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