import matplotlib.pyplot as plt
import csv
import numpy as np
from utils import RAW_CLASSES, load_feature_data_together
from sklearn.cluster import KMeans

def clustering(c_names=[], data=[], a_infos=[], save_path=""):
    c_names = np.array(c_names)
    data = np.array(data)
    a_infos = np.array(a_infos)
    kmeans = KMeans(n_clusters=12, random_state=0).fit(data)
    labels = kmeans.labels_
    print(labels)

    clustered_comp = []
    clustered_action = []
    clustered_data = []
    for l in range(np.max(labels)+1):
        inds = labels==l
        l_data = data[inds, :]
        clustered_data += l_data.tolist()
        clustered_data.append(np.ones(22))
        clustered_comp += c_names[inds].tolist()
        clustered_comp.append("")
        clustered_action += a_infos[inds, :].tolist()
        clustered_action.append(["", ""])

    clustered_c_a = []
    for c, a in zip(clustered_comp, clustered_action):
        clustered_c_a.append(c+"_"+a[1])

    fig, ax = plt.subplots()
    im = ax.imshow(np.transpose(clustered_data))
    print(len(clustered_c_a), len(clustered_data))

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(clustered_c_a)), labels=clustered_c_a)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title("kmeans")
    fig.tight_layout()
    plt.show()
    print(clustered_comp, clustered_action) #, clustered_data)



if __name__ == "__main__":
    compound_names, all_median_data, action_information = \
        load_feature_data_together(path="/srv/yanke/PycharmProjects/HTScreening/data/median/median_compounds_feature_fish_with_action.csv")
    clustering(compound_names, all_median_data, action_information, save_path="/srv/yanke/PycharmProjects/HTScreening/data/median/")
