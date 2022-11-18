import numpy as np
import nthresh
import csv
import sys
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

from scipy.stats import ttest_ind

from utils import read_mode_action
from config import *

"""
visualize the intergrated feature
"""
p_thre=0.05

def read_binary_code_patterns(path, save_path):
    comp_names = []
    data = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            compound_name = l[0]
            comp_names.append(compound_name)
            data.append([int(x) for x in l[1:-2]])

    data = np.array(data)
    comp_names = np.array(comp_names)
    print(data)
    return comp_names, data

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    print(model.children_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def HI_clustering(path):
    data = pd.read_csv(path, usecols=["Compound", "Phase 1", "Phase 2", "Phase 3"])
    data.set_index(['Compound'], drop=True, inplace=True)
    data.index.name = None

    print(data)
    #fig = plt.figure(figsize=(15, 20))
    heatmap = sns.clustermap(data=data, method='ward', metric='euclidean',
                             row_cluster=True, col_cluster=None, cmap="coolwarm",
                             vmin=0, vmax=2, figsize=(15, 55))
    heatmap.fig.suptitle("Hierarchy Clustering", fontsize=20)
    heatmap.ax_heatmap.set_title("Hierarchical Clustering(ward)", fontsize=20)
    #plt.show()
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(SAVE_FINAL_RESULT_PATH / ("hierarchical_clustering_with_binary_codes_p" + str(p_thre) + ".png"), dpi=300)

    """
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(data)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    """

if __name__ == "__main__":
    #save_binary_code_mapping_motion(
    #    binary_path="/Users/yankeewann/Desktop/HTScreening/data/featured/effects_binary_codes_with_integration.csv",
    #    save_path="/Users/yankeewann/Desktop/HTScreening/data/")

    #comp_names, data = read_binary_code_patterns(SAVE_FEATURE_PATH / ("effects_binary_codes_with_integration" + str(p_thre)+".csv"), DATA_PATH)
    HI_clustering(SAVE_FEATURE_PATH / ("effects_binary_codes_with_integration" + str(p_thre)+".csv"))

