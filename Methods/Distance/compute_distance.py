import numpy as np
import csv
from dataloader import load_cleaned_data, load_featured_data
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

"""
(x-mu)^T * _Sigma^(-1) * (x-mu)
"""

def compute_distance(mu, sigma_inv, data):
    return np.matmul(np.matmul(np.transpose(data - mu), sigma_inv), (data - mu))

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



if __name__ == "__main__":
    #data, _, lablels, actions = load_cleaned_data(
    #    path="/Users/yankeewann/Desktop/HTScreening/data/cleaned/all_compounds_ori_fish_with_action.csv")
    data, _, lablels, actions = load_featured_data(
        path="/Users/yankeewann/Desktop/HTScreening/data/featured/all_compounds_feature_max_median_fish_with_action.csv")
    save_path = "/Users/yankeewann/Desktop/HTScreening/data/distance/"
    distance_to(data, lablels, save_path)