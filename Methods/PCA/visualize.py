import pickle
import matplotlib.pyplot as plt
from PCA import PCA_torch
import os
import csv
import numpy as np
from pca_dim_reduce import try_pca_by_compounds, try_pca_by_compounds_with_clustering
from data_loader import  load_feature_data_together, load_data
from pca_dim_reduce import try_PCA_with_torch
from sklearn.cluster import KMeans
from PCA import PCA_torch
#im = pickle.load(open("./results/pca_with_name/train/VAE_embedding.pickle", "rb"))
#plt.show()

def visualize_compound_after_PCA(compounds=[], input_dimension = 541):
    PCA_dim = 50
    base_path = "/srv/yanke/PycharmProjects/HTScreening/data/raw_data/old_compounds/"
    pca = PCA_torch(center=False, n_components=PCA_dim)


    all_data = []
    data_dict = {}
    data_files = os.listdir(base_path)
    for d_f in data_files:
        d_f_name = d_f[:-6]
        if d_f_name[:2] == "WT":
            d_f_name = "WT"
        # print(d_f_name, lb)
        data_dict[d_f_name] = []
        d_l = []
        with open(base_path + d_f, newline='') as csv_f:
            read_lines = csv.reader(csv_f, delimiter=",")
            for j, l in enumerate(read_lines):
                if j > 1:
                    data_line = [float(i) for i in l]
                    d_l.append(data_line)
            d_l = np.array(d_l, np.float)
            for i in range(d_l.shape[1]):
                # print(list(d_l[:, i]))
                data_item = list(d_l[:input_dimension, i])
                data_dict[d_f_name].append(data_item)
                all_data.append(data_item)

    all_data = np.array(all_data)
    new_all_train = pca.fit_PCA(all_data)

    #dis_num = 0
    #for c in compounds:
    #    dis_num += len(data_dict[c])

    display_data = []#np.zeros((dis_num, 10), np.float)
    #fig, axs = plt.subplots(len(compounds)+1)
    fig = plt.figure()
    for i in range(len(compounds)):
        comp_data = data_dict[compounds[i]]
        comp_data = np.array(comp_data)
        new_comp_data = pca.test(comp_data)
        for d in range(new_comp_data.shape[0]):
            display_data.append(new_comp_data[d])
        display_data.append(np.zeros((PCA_dim), dtype=np.float))
        #axs[i].imshow(new_comp_data, cmap="hot", interpolation="nearest")
        #axs[i].set_ylabel("fish case")
        #axs[i].set_xlabel("dimension")
        #axs[i].set_title(compounds[i])
        #plt.plot(plot_data[i, :])
    #plt.xlabel("time")
    #plt.title("motion index")
    #axs[-1].imshow(np.array(display_data), cmap="hot", interpolation="nearest")
    #axs[-1].set_ylabel("fish case")
    #axs[-1].set_xlabel("dimension")
    #axs[-1].set_title("all")
    im = plt.imshow(np.array(display_data), cmap="gray", interpolation="nearest")
    plt.colorbar(im)
    plt.ylabel("Fish case")
    plt.xlabel("Components of PCA")
    plt.title("all")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()


def visualize_compound_cleaned_after_PCA(compounds=[], dim_begin = 0, dim_end = 188):
    data_dict, all_data = load_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/", dim_begin, dim_end)
    display_data = try_pca_by_compounds(pca_dim=50, compounds=compounds, all_data=all_data, data_dict=data_dict)

    im = plt.imshow(np.array(display_data), cmap="gray", interpolation="nearest")
    plt.colorbar(im)
    plt.ylabel("Fish case")
    plt.xlabel("Components of PCA")
    plt.title("all")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()

def visualize_compound_cleaned_after_PCA_clustering(compounds=[], dim_begin = 0, dim_end = 188):
    data_dict, all_data = load_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_data/", dim_begin, dim_end)
    display_data = try_pca_by_compounds_with_clustering(pca_dim=50, compounds=compounds, all_data=all_data, data_dict=data_dict)

    im = plt.imshow(np.array(display_data), cmap="cool", interpolation="nearest")
    plt.colorbar(im)
    plt.ylabel("Fish case")
    plt.xlabel("Components of PCA")
    plt.title("all")
    plt.tight_layout()
    plt.show()
    #max_v = np.max(plot_data)
    #plot_data /= max_v
    #ax = sns.heatmap(plot_data, linewidth=0.5)
    #plt.show()

def visualize_compound_median_after_PCA():
    compound_names, all_median_data, action_information = \
        load_feature_data_together(
            path="/Users/yankeewann/Desktop/HTScreening/data/median/median_compounds_ori_fish_with_action.csv")
    c_names = np.array(compound_names)
    data = np.array(all_median_data)
    a_infos = np.array(action_information)

    pca = PCA_torch(center=False, n_components=2)
    new_train = pca.fit_PCA(data)


    kmeans = KMeans(n_clusters=1, random_state=0).fit(data)
    labels = kmeans.labels_
    print(labels)

    colormap = plt.cm.Dark2.colors
    for l, color in zip(range(np.max(labels) + 1), colormap):
        inds = labels == l
        l_data = data[inds, :]
        new_l_data = pca.test(l_data)

        for d, c, a in zip(new_l_data, c_names[inds].tolist(), a_infos[inds, :].tolist()):
            plt.scatter(d[0], d[1], s=15, color=color)
            plt.text(d[0], d[1], c + "_" + a[1], fontsize=6)
    plt.text(0.4, -0.2, "median of\noriginal time-series data\n+ PCA", fontsize=10) #"1st + 3rd quantiles and median"
    plt.ylabel("feature 1")
    plt.xlabel("feature 2")
    plt.title("PCA of median of original time-series data of each compound")
    plt.tight_layout()
    plt.show()
    plt.clean()

if __name__ == "__main__":
    #visualize(path="./data/raw_data/old_compounds/")
    #trainset = RawDataSet(path="./data/raw_data/old_compounds/", label_path="./data/data_median_all_label.csv")
    #visualize_PCA()
    #visualize_compound_cleaned_after_PCA(["WT", "C5", "C12", "C88", "C105", "C117"], dim_begin=0,
    #                                              dim_end=541)
    #visualize_compound_cleaned_after_PCA_clustering(["WT", "C5", "C105"], dim_begin = 0, dim_end = 541)# ["C6", "C5", "C12", "C88", "C105", "C117"], dim_begin = 0, dim_end = 541) #, "C14", "C20", "C88"]) # 0, 188, 364, 541
    #visualize_compound_after_PCA(["WT", "C5", "C12", "C88", "C105", "C117"])
    visualize_compound_median_after_PCA()