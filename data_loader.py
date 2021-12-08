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
        data_files = os.listdir(path)
        for d_f in data_files:
            read_lines = csv.reader(path + d_f)
            for l in read_lines:
                data_files.append(l)

        self.data = data_list

    def __getitem__(self, index):

        d = torch.from_numpy(self.data[index]).float()

        return d

    def __len__(self):
        return len(self.data)