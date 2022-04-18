import os
import math
import csv
import sys
from skimage import io, draw, color, transform
import numpy as np
import matplotlib.pyplot as plt

def get_wt_data(path):
    all_data = {}
    wt_folders = os.listdir(path)
    for w_folder in wt_folders:
        all_data[w_folder] = []
        wt_files = os.listdir(path+w_folder+"/")
        for w_f in wt_files:
            with open(path+w_folder+"/"+w_f, newline='') as csv_f:
                read_lines = csv.reader(csv_f, delimiter=",")
                wt_d_l = []
                for j, l in enumerate(read_lines):
                    if j > 0:
                        data_line = [float(i) for i in l]
                        wt_d_l.append(data_line)
                wt_d_l = np.array(wt_d_l, np.float)
                print(wt_d_l.shape)

                for i in range(wt_d_l.shape[1]):
                    # print(list(d_l[:, i]))
                    all_data[w_folder].append(list(wt_d_l[:, i]))

    return all_data

def visualize(data_dict={}):
    wt_types = data_dict.keys()
    fig, axs = plt.subplots(len(wt_types))
    for w, wt in enumerate(wt_types):
        print(w, wt)
        num = len(data_dict[wt])
        for w_d in data_dict[wt]:
            axs[w].plot(w_d)
        axs[w].set_ylabel("motion index")
        axs[w].set_xlabel("time")
        axs[w].set_title(wt + "("+str(num)+")")
        axs[w].margins(x=0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    wt_data = get_wt_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned/all_data/Controls/")
    visualize(wt_data)
    #split_data(datas, labels, save_path="./data/effected_dataset/")