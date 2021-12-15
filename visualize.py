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
from data_loader import RawDataSet

def visualize(path):
    data_lists=[]
    data_files = os.listdir(path)
    for d_f in data_files:
        print(d_f)
        data_list = []
        with open(path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter = ",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    data_list.append(data_line)
        data_lists.append(data_list)

    for d_l in data_lists:
        d_l = np.array(d_l)
        for i in range(d_l.shape[1]):
            plt.plot(d_l[:, i])
        plt.show()



if __name__ == "__main__":
    #visualize(path="./data/raw_data/old_compounds/")
    trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv")