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

def compute_distance(mu, sigma_inv, data):
    return np.matmul(np.matmul(np.transpose(data - mu), sigma_inv), (data - mu))

def compute_mean_distance(mu1, mu2, sigma1, sigma2):
    sigma = 0.5 * (sigma1 + sigma2)
    sigma_inv = np.linalg.inv(sigma)
    return np.matmul(np.matmul(np.transpose(mu1 - mu2), sigma_inv), (mu1 - mu2))

def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)

def stdev(data):
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev

def diff(data):
    data_1 = np.array(data[1:])
    diff = data_1 - np.array(data[:-1])
    return diff

def deriv(x, y):
    dy_dx = diff(y) / diff(x)
    return np.array(x[:-1]), dy_dx

def draw_distance_figure(all_dis, dis_save, max_v, min_v, thresholds):
    num = len(dis_save)
    #max_v = np.max(np.array(dis_data)[:, -1])

    WILD_dis = all_dis[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-1, num+2)
    ax.set_ylim(min_v-200, max_v+200)

    x_labels = []
    ax.axhline(y=0, color="pink")

    for n, (a_d, d) in enumerate(zip(all_dis, dis_save)):
        ax.axvline(x=n, ymin=0, ymax=max_v, color="grey")
        #ax.add_patch(Rectangle((n - 0.25, d[2]), 0.5, d[3] - d[2])) #quantile
        ax.add_patch(Rectangle((n-0.25, d[1]-d[4]), 0.5, d[4]*2)) #std
        ax.scatter(num, d[1], s=2.5, c='blue')
        ax.add_patch(Rectangle((n-0.25, d[1]-2), 0.5, 4,
                               facecolor='black',
                               fill=True))
        F, p = ttest_ind(WILD_dis, a_d, equal_var=False)
        #ax.scatter(np.ones(len(all_dis[n]))*n, all_dis[n], s=2.5, c='blue')
        #ax.text(n - 0.2, d[3] + 80, "3rd:" + str(round(d[3], 2)), fontsize=8)
        #ax.text(n - 0.2, d[3] + 40, "$\mu$:" + str(round(d[1], 2)), fontsize=8)
        #ax.text(n - 0.2, d[3] + 20, "$\sigma$:" + str(round(d[4], 2)), fontsize=8)
        ax.text(n - 0.5, d[1] + d[4] + 80, str(round(p, 3)), fontsize=8)
        x_labels.append(d[0])
    x_labels.append("All")

    """
    # draw the threshold
    for t in thresholds:
        ax.axhline(y=t, color="red", ls="--", lw=0.1)
        ax.text(num-0.5, t, "Thre:" + str(round(t, 2)), fontsize=5, color="red")
    """

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Rotate the tick labels and set their alignment.
    ax.set_ylabel("Distance")
    ax.text(0, max_v+80, "Distance = $\sqrt[2]{(x-\mu_w)^T * \Sigma^{-1}_w * (x-\mu_w)}$", bbox=dict(facecolor='red', alpha=0.5))
    plt.tight_layout()
    plt.show()

def draw_quantiles(data):
    """
    x_s = []
    quantiles = []
    for i in range(101):
        x_s.append(i/100)
        q = np.quantile(data, i/100)
        print(i/100, q)
        quantiles.append(q)


    x, dy_dx = deriv(quantiles, x_s)
    fig, ax = plt.subplots()
    #hist, _ = np.histogram(data, bins=100)
    ax.hist(data, bins=100)
    ax.plot(quantiles, x_s, c='blue')
    """
    (mu, sigma) = norm.fit(data)
    a=4
    ae, loce, scalee = skewnorm.fit(data)
    # the histogram of the data

    n, bins, patches = plt.hist(data, 100, density=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    p = skewnorm.pdf(bins, ae, loce, scalee)  # .rvs(100)
    #plt.plot(x, p, 'k', linewidth=2)
    #first_quantile = np.quantile(data, 0.25)

    #print(first_quantile)
    #plt.axvline(np.quantile(data, 0.25), ymin=0, ymax = 0.25, linewidth=2, color='r')
    #plt.axvline(np.quantile(data, 0.75), ymin=0, ymax=0.25, linewidth=2, color='r')
    l = plt.plot(bins, p, 'x', linewidth=2, label="Skew Norm")
    l = plt.plot(bins, y, 'r--', linewidth=2, c="black", label = "Norm")
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'$Norm: \mu=%.3f, \sigma=%.3f;'
              r' Skew norm: \mu_k=%.3f, \sigma_k=%.3f$' % (mu, sigma, loce, scalee))
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()
    """
    ax.set_xticks(np.arange(len(x_s)), labels=x_s)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    """
    # Rotate the tick labels and set their alignment.
    #ax.set_ylabel("Distance")
    #plt.tight_layout()
    #plt.show()

def fit_skew_dist(data):
    #fig, ax = plt.subplots(1, 1)
    a = 4
    mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')

    n, bins, patches = plt.hist(data, 60, density=1, facecolor='green', alpha=0.75)
    y = skewnorm.pdf(bins, a, mean, var)
    plt.plot(bins, y, 'r--', linewidth=2)

    plt.show()


def distance_to(data, labels, save_path):
    label_num = np.max(labels)
    wild_data = data[labels==0]
    WILD_MEAN = np.mean(wild_data, axis=0)
    WILD_COV_INV = np.linalg.inv(np.cov(wild_data, rowvar=False))
    all_distance_save = []

    all_dis = []
    max_dis = []
    min_dis = []
    for l in range(label_num+1):
        inds = labels==l
        comp_data = data[inds]
        if comp_data.shape[0] < 1:
            continue
        comp_dis = []
        for c_d_i in range(comp_data.shape[0]):
            dis = compute_distance(WILD_MEAN, WILD_COV_INV, comp_data[c_d_i, :])
            #dis = np.linalg.norm(WILD_MEAN - comp_data[c_d_i, :])
            #print(WILD_MEAN.shape, comp_data[c_d_i, :].shape)
            #print(dis)
            dis = np.sqrt(dis)
            #print(dis)
            comp_dis.append(dis)
        #if l==0:
        #    print(comp_dis)
        all_dis.append(comp_dis)

        std_v = stdev(comp_dis)

        comp_dis = np.array(comp_dis)
        #draw_quantiles(comp_dis)
        #fit_skew_dist(comp_data)
        #print(comp_dis)
        l_q = np.quantile(comp_dis, 0.25) #np.min(comp_dis)
        h_q = np.quantile(comp_dis, 0.75)  #np.max(comp_dis)
        mean_v = comp_dis.mean() #np.mean(comp_dis)

        feature_data = ["C"+str(l)]
        feature_data.append(mean_v)
        feature_data.append(l_q)
        feature_data.append(h_q)
        feature_data.append(std_v)
        all_distance_save.append(feature_data)
        max_dis.append(mean_v+std_v)
        min_dis.append(mean_v-std_v)
    max_v = np.max(max_dis)
    min_v = np.min(min_dis)
    print("max is ", np.argmax(max_dis), max_v)
    print("min is ", np.argmin(min_dis), min_v)
    #thre = nthresh.nthresh(np.array(all_dis), n_classes=4, bins=10, n_jobs=1)
    thre = [all_distance_save[0][1] + 3*all_distance_save[0][4]]#[24.81]
    print("the threshold is ", thre)

    draw_distance_figure(all_dis, all_distance_save, max_v, min_v, thre)
    with open(save_path + "all_compounds_distance_to_wild_ori.csv", "w") as save_csv:
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
    #    path="/Users/yankeewann/Desktop/HTScreening/data/featured/all_compounds_feature_max_median_fish_with_action.csv")
    save_path = "/Users/yankeewann/Desktop/HTScreening/data/distance/"
    distance_to(data, labels, save_path)
    #mean_distance_with_PCA_visualize(data, labels)
