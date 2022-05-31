import numpy as np
import csv
import sys
sys.path.append("../../")
from Methods.Distance.dataloader import load_cleaned_data, load_featured_data
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from Methods.PCA.PCA import PCA_torch
"""
(x-mu)^T * _Sigma^(-1) * (x-mu)
"""

def compute_distance(mu, sigma_inv, data):
    return np.matmul(np.matmul(np.transpose(data - mu), sigma_inv), (data - mu))

def compute_mean_distance(mu1, mu2, sigma1, sigma2):
    sigma = 0.5 * (sigma1 + sigma2)
    sigma_inv = np.linalg.inv(sigma)
    return np.matmul(np.matmul(np.transpose(mu1 - mu2), sigma_inv), (mu1 - mu2))

def draw_distance_figure(dis_data, max_v):
    num = len(dis_data)
    #max_v = np.max(np.array(dis_data)[:, -1])
    fig, ax = plt.subplots()
    ax.set_xlim(-1, num+1)
    ax.set_ylim(-1, max_v+1)

    x_labels = []
    ax.axhline(y=0, color="pink")
    for n, d in enumerate(dis_data):
        ax.axvline(x=n, ymin=0, ymax=max_v, color="grey")
        ax.add_patch(Rectangle((n-0.25, d[2]), 0.5, d[3]-d[2]))
        ax.add_patch(Rectangle((n-0.25, d[1]), 0.5, 0.1,
                               facecolor='pink',
                               fill=True))
        x_labels.append(d[0])
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Rotate the tick labels and set their alignment.
    ax.set_ylabel("Distance")
    ax.text(4, 10, "Distance = $\sqrt[2]{(x-\mu_w)^T * \Sigma^{-1}_w * (x-\mu_w)}$", bbox=dict(facecolor='red', alpha=0.5))
    plt.tight_layout()
    plt.show()

def distance_to(data, labels, save_path):
    label_num = np.max(labels)
    wild_data = data[labels==0]
    WILD_MEAN = np.mean(wild_data, axis=0)
    WILD_COV_INV = np.linalg.inv(np.cov(wild_data, rowvar=False))
    all_distance_save = []
    all_dis = []
    for l in range(label_num+1):
        inds = labels==l
        comp_data = data[inds]
        if comp_data.shape[0] < 1:
            continue
        comp_dis = []
        for c_d_i in range(comp_data.shape[0]):
            dis = compute_distance(WILD_MEAN, WILD_COV_INV, comp_data[c_d_i, :])
            #print(dis)
            dis = np.sqrt(dis)
            #print(dis)
            comp_dis.append(dis)
        comp_dis = np.array(comp_dis)
        #print(comp_dis)
        l_q = np.quantile(comp_dis, 0.25)
        h_q = np.quantile(comp_dis, 0.75)
        median_v = np.median(comp_dis)
        feature_data = ["C"+str(l)]
        feature_data.append(median_v)
        feature_data.append(l_q)
        feature_data.append(h_q)
        all_distance_save.append(feature_data)
        all_dis.append(h_q)
    max_v = np.max(all_dis)
    draw_distance_figure(all_distance_save, max_v)
    with open(save_path + "all_compounds_distance_to_wild_featured.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_distance_save)

def mean_distance_with_PCA_visualize(data, labels):
    pca = PCA_torch(center=False, n_components=2)

    data_mean = []
    label_num = np.max(labels)
    wild_data = data[labels==0]
    WILD_MEAN = np.mean(wild_data, axis=0)
    data_mean.append(WILD_MEAN)
    WILD_COV = np.cov(wild_data, rowvar=False)
    all_distance_save = []
    all_dis = []
    all_dis.append(0) # distance for wild type
    label_strs = []
    label_strs.append("C" + str(0))
    for l in range(label_num+1):
        inds = labels==l
        comp_data = data[inds]
        if comp_data.shape[0] < 1:
            continue
        comp_mean = np.mean(comp_data, axis=0)
        data_mean.append(comp_mean)
        comp_cov = np.cov(comp_data, rowvar=False)
        comp_dis = compute_mean_distance(comp_mean, WILD_MEAN, comp_cov, WILD_COV)
        comp_dis = np.sqrt(comp_dis)
        #print(comp_dis)
        feature_data = ["C"+str(l)]
        label_strs.append("C"+str(l))
        feature_data.append(comp_dis)
        all_distance_save.append(feature_data)
        all_dis.append(comp_dis)
    data_mean = np.array(data_mean)

    _ = pca.fit_PCA(data_mean)
    new_data_mean = pca.test(data_mean)
    plt.scatter(new_data_mean[:, 0], new_data_mean[:, 1], s=15, color="blue")
    plt.text(new_data_mean[0, 0], new_data_mean[0, 1], label_strs[0], fontsize=6)
    for d, p, l_s in zip(all_dis[1:], new_data_mean[1:, :], label_strs[1:]):
        plt.plot([new_data_mean[0, 0], p[0]], [new_data_mean[0, 1], p[1]],
                 color="grey", linestyle='-', linewidth=0.1)
        plt.text(new_data_mean[0, 0]*0.2 + p[0]*0.8, new_data_mean[0, 1]*0.2 + p[1]*0.8, round(d, 2), fontsize=6)
        plt.text(p[0], p[1], l_s, fontsize=6)
    plt.text(0.4, 0.2,
             "Distance = $\sqrt[2]{(\mu_c-\mu_w)^T * [(\Sigma_c + \Sigma_w) / 2]^{-1} * (\mu_c-\mu_w)}$",
             bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0.4, 0.4,
             "1) Mean of time-series data of each compound\n2) PCA of the means of all compounds" ,
             bbox=dict(facecolor='red', alpha=0.5))
    plt.ylabel("feature 1")
    plt.xlabel("feature 2")
    plt.title("PCA of the mean of the time-series data for each compound")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data, _, labels, actions = load_cleaned_data(
        path="/Users/yankeewann/Desktop/HTScreening/data/cleaned/all_compounds_ori_fish_with_action.csv")
    #data, _, labels, actions = load_featured_data(
    #    path="/Users/yankeewann/Desktop/HTScreening/data/featured/all.....csv")
    save_path = "/Users/yankeewann/Desktop/HTScreening/data/distance/"
    #distance_to(data, labels, save_path)
    mean_distance_with_PCA_visualize(data, labels)
