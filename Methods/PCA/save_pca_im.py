import pickle
import matplotlib.pyplot as plt
from PCA import PCA_torch
import os
import csv
import numpy as np
from pca_dim_reduce import save_im_pca_by_compounds_with_clustering
from utils import load_data


# im = pickle.load(open("./results/pca_with_name/train/VAE_embedding.pickle", "rb"))
# plt.show()

if __name__ == "__main__":
    #visualize(path="./data/raw_data/old_compounds/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv")
    #visualize_PCA()
    #visualize_compound_cleaned_after_PCA(["WT", "C5", "C12", "C88", "C105", "C117"], dim_begin=0,
    #                                              dim_end=541)
    data_dict, all_data = load_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/", 0, 541)
    save_im_pca_by_compounds_with_clustering(pca_dim=32, all_data=all_data, data_dict=data_dict, save_path="./results/pca_images/")