import numpy as np
import nthresh
import csv
import sys
sys.path.append("../../")
import math
from scipy.stats import norm, skewnorm
import matplotlib.mlab as mlab
from Methods.Distance.dataloader import load_cleaned_data, load_featured_data
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind

from Methods.PCA.PCA import PCA_torch
"""
(x-mu)^T * _Sigma^(-1) * (x-mu)
"""

def save_figure_to(data, labels, save_path):
    label_num = np.max(labels)
    for l in range(label_num+1):
        inds = labels==l
        comp_data = data[inds]
        if comp_data.shape[0] < 1:
            continue
        plt.figure(figsize=(30,5))
        for c_d in range(comp_data.shape[0]):
            plt.plot(comp_data[c_d, :])
        plt.title("C_" + str(l))
        plt.tight_layout()
        plt.savefig(save_path + "C_" + str(l) + ".png")
        plt.clf()

if __name__ == "__main__":
    data, _, labels, actions = load_cleaned_data(
        path="/Users/yankeewann/Desktop/HTScreening/data/cleaned/all_compounds_ori_fish_with_action.csv")
    #data, _, labels, actions = load_featured_data(
    #   path="/Users/yankeewann/Desktop/HTScreening/data/featured/all_compounds_pca_feature_fish_with_action.csv")
    save_path = "/Users/yankeewann/Desktop/HTScreening/data/cleaned/plots/"
    save_figure_to(data, labels, save_path)
    #mean_distance_with_PCA_visualize(data, labels)
