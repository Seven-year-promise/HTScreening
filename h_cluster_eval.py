import numpy as np
import csv
import sys
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

from scipy.stats import ttest_ind

from utils import read_mode_action
from config import *

feature_num = 7
algorithm = "wasserstein_2" # hellinger, wasserstein
type = "integration"
control_name = "all"

def eval_tree_diversity(compound_list, labels):
    diversity = []
    compound_list = np.array(compound_list)
    for l in list(set(labels)):
        all_compounds_for_l = compound_list[[i for i, val in enumerate(labels) if val==l]]
        #print(l, all_compounds__for_l)
        all_actions_for_i = []
        for c in all_compounds_for_l:
            all_actions_for_i.append(c.split("_")[1])
        #print(all_actions_for_i)
        diversity.append(len(list(set(all_actions_for_i))))
    return np.average(diversity)

def prediction_effects(compound_list, labels, cloest_comp):

    label = labels[compound_list.index(cloest_comp)]
    #print(label, labels)
    compound_list = np.array(compound_list)
    all_compounds_for_l = compound_list[[i for i, val in enumerate(labels) if val == label]]


    final_effect = []
    for c in all_compounds_for_l:
        final_effect.append(c.split("_")[1])
    final_effect = list(set(final_effect))
    print(final_effect)
    return final_effect

def load_train_model(link_path, data_path, test_data_path, thre_intervel=100):
    link = pd.read_csv(link_path,  usecols=["0", "1", "2", "3"])
    max_eucli_dis = np.max(link["2"])
    link = link.to_numpy()

    data = pd.read_csv(data_path, usecols=["Compound"] + ["Feature " + str(x) for x in range(feature_num)] + ["Action"])

    data['Compound'] += "_" + data['Action']
    data.set_index(['Compound'], drop=True, inplace=True)
    data = data.drop('Action', axis=1)
    data.index.name = None

    test_data = pd.read_csv(test_data_path, usecols=["Compound"] + ["Feature " + str(x) for x in range(feature_num)] + ["Action"])
    test_labels = test_data['Action']
    test_names = test_data['Compound']

    # normalize
    for f_name in data.columns[1:-1]:
        max_value = data[f_name].max()
        min_value = data[f_name].min()
        data[f_name] = (data[f_name] - min_value) / (max_value - min_value)
        data[f_name] = data[f_name] * 2 - 1

        test_data[f_name] = (test_data[f_name] - min_value) / (max_value - min_value)
        test_data[f_name] = test_data[f_name] * 2 - 1

    test_data.set_index(['Compound'], drop=True, inplace=True)
    test_data = test_data.drop('Action', axis=1)
    test_data.index.name = None

    #test_data.set_index(['Compound'], drop=True, inplace=True)
    #test_data = test_data.drop('Action', axis=1)
    #test_data.index.name = None

    for thre in range(thre_intervel+1):
        thre = 10
        t = thre/thre_intervel*max_eucli_dis

        #print(new_data, new_data.index)
        B = dendrogram(link, labels=list(data.index), color_threshold=t)
        #print(B['leaves_color_list'], B['ivl']) # NOTE  B['ivl']: the ordered leaves,  B['leaves_color_list']: labels
        diversity_h_cluster = eval_tree_diversity(compound_list=B['ivl'], labels=B['leaves_color_list'])
        #print(diversity_h_cluster)
        predictions = []
        for index, test_d in test_data.T.iteritems():
            dis = (data - list(test_d)).pow(2).sum(1).pow(0.5)
            cloest_comp = dis.idxmin()
            preds = prediction_effects(compound_list=B['ivl'], labels=B['leaves_color_list'], cloest_comp=cloest_comp)
            predictions.append(preds)

        #print(predictions)
        plt.show()

        """
        for i in test_names:
            t_d = np.array(test_data.loc[i])
            data_t_dist = pdist(data_t)
            break
        """
        break



if __name__ == "__main__":
    #save_binary_code_mapping_motion(
    #    binary_path="/Users/yankeewann/Desktop/HTScreening/data/featured/effects_binary_codes_with_integration.csv",
    #    save_path="/Users/yankeewann/Desktop/HTScreening/data/")

    #comp_names, data = read_binary_code_patterns(SAVE_FEATURE_PATH / ("effects_binary_codes_with_integration" + str(p_thre)+".csv"), DATA_PATH)
    load_train_model(link_path=TRAIN_SAVE_FINAL_RESULT_PATH / ("hierarchical_clustering_with_"+str(feature_num)+type+"_distance_"+control_name+"_linkage.csv"),
                     data_path=TRAIN_SAVE_FEATURE_PATH / ("effects_distance_with_"+str(feature_num)+type+"_"+algorithm+"_"+control_name+".csv"),
                     test_data_path=TEST_SAVE_FEATURE_PATH / ("effects_distance_with_"+str(feature_num)+type+"_"+algorithm+"_"+control_name+".csv"),
                     thre_intervel=100)
