import numpy as np
import nthresh
import csv
import sys
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
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
feature_num = 14 # 14
type = "quartile"  # "integration"
control_name = "all"

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

def get_clusters(heatmap_sns):
    print(heatmap_sns.dendrogram_row.reordered_ind)
    den = scipy.cluster.hierarchy.dendrogram(heatmap_sns.dendrogram_col.linkage,
                                             labels=df.index,
                                             color_threshold=0.60)
    from collections import defaultdict

    def get_cluster_classes(den, label='ivl'):
        cluster_idxs = defaultdict(list)
        for c, pi in zip(den['color_list'], den['icoord']):
            for leg in pi[1:3]:
                i = (leg - 5.0) / 10.0
                if abs(i - int(i)) < 1e-5:
                    cluster_idxs[c].append(int(i))

        cluster_classes = {}
        for c, l in cluster_idxs.items():
            i_l = [den[label][i] for i in l]
            cluster_classes[c] = i_l

        return cluster_classes

    clusters = get_cluster_classes(den)

    #print(clusters)
    """
    cluster = []
    for i in df.index:
        included = False
        for j in clusters.keys():
            if i in clusters[j]:
                cluster.append(j)
                included = True
        if not included:
            cluster.append(None)

    df["cluster"] = cluster
    
    """


def extract_clustered_table(res, data):
    """
    input
    =====
    res:       the clustermap object
    data:                input table

    output
    ======
    returns:             reordered input table
    """

    # if sns.clustermap is run with row_cluster=False:
    if res.dendrogram_row is None:
        print("Apparently, rows were not clustered.")
        return -1

    if res.dendrogram_col is not None:
        # reordering index and columns
        new_cols = data.columns[res.dendrogram_col.reordered_ind]
        new_ind = data.index[res.dendrogram_row.reordered_ind]

        return data.loc[new_ind, new_cols]

    else:
        # reordering the index
        new_ind = data.index[res.dendrogram_row.reordered_ind]

        return data.loc[new_ind, :]

def HI_clustering(path):
    data = pd.read_csv(path, usecols=["Compound"] + ["Feature " + str(x) for x in range(feature_num)]+["Action"])
    data['Compound'] += "_" + data['Action']

    data.set_index(['Compound'], drop=True, inplace=True)
    data = data.drop('Action', axis=1)
    data.index.name = None

    #fig = plt.figure(figsize=(15, 20))
    heatmap = sns.clustermap(data=data, method='ward', metric='euclidean',
                             row_cluster=True, col_cluster=None, cmap="coolwarm",
                             vmin=0, vmax=2, figsize=(15, 55))
    ordered_data = extract_clustered_table(heatmap, data)

    heatmap.fig.suptitle("Hierarchy Clustering", fontsize=20)
    heatmap.ax_heatmap.set_title("Hierarchical Clustering(ward)", fontsize=20)
    #plt.show()
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(SAVE_FINAL_RESULT_PATH / ("hierarchical_clustering_with_"+str(feature_num) + type+"_binary_codes_p" + str(p_thre) + ".png"), dpi=300)




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
    HI_clustering(SAVE_FEATURE_PATH / ("effects_binary_codes_with_"+str(feature_num)+type + str(p_thre)+control_name+".csv"))

