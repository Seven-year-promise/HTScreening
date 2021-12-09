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

class DataSet(data.Dataset):
    def __init__(self, path):
        data_list = []
        data_labels = []
        data_files = os.listdir(path)
        for d_f in data_files:
            with open(path + d_f, newline='') as csv_f:
                read_lines = csv.reader(csv_f, delimiter = ",")
                for j, l in enumerate(read_lines):
                    if j > 0:
                        data_line = [float(i) for i in l]
                        data_list.append(data_line)
                        print(data_line)

        self.data = data_list
        self.labels = data_labels

    def __getitem__(self, index):

        d = torch.from_numpy(self.data[index]).float()
        l = torch.from_numpy(self.labels[index]).long()
        return d, l

    def __len__(self):
        return len(self.data)