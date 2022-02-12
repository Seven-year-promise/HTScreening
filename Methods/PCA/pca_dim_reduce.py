import sys
#sys.path.insert(0, "/srv/yanke/PycharmProjects/HTScreening")
import numpy as np
import cv2
import sys
sys.path.append("../../")
import csv
from Methods.PCA.PCA import PCA_torch
from Methods.PCA.plot_tsne import plot_dist_no_label, plot_dist_with_label, plot_dist_name, plot_dist_train_test, plot_tsne_train_test
from sklearn.cluster import MeanShift, KMeans
from Methods.PCA.utils import load_train_test_data, load_cleaned_train_test_data

def try_PCA_with_torch(data):
    pca = PCA_torch(center=False, n_components=2)
    new_feature = pca.fit_PCA(data)

    return data, new_feature

def try_PCA_with_test(train_set, test_set):
    pca = PCA_torch(center=False, n_components=2)
    new_train = pca.fit_PCA(train_set)
    new_test = pca.test(test_set)

    return new_train, new_test

def try_pca_by_compounds(pca_dim, compounds, all_data, data_dict):
    PCA_dim = pca_dim
    pca = PCA_torch(center=False, n_components=PCA_dim)
    all_data = np.array(all_data)
    print("number of all data: ", all_data.shape[0])
    _ = pca.fit_PCA(all_data)

    # dis_num = 0
    # for c in compounds:
    #    dis_num += len(data_dict[c])

    display_data = []  # np.zeros((dis_num, 10), np.float)
    # fig, axs = plt.subplots(len(compounds)+1)
    for i in range(len(compounds)):

        comp_data = data_dict[compounds[i]]
        num_data = len(comp_data)
        print("number of data for compound " + compounds[i], num_data)
        comp_data = np.array(comp_data)
        new_comp_data = pca.test(comp_data)
        for d in range(new_comp_data.shape[0]):
            display_data.append(new_comp_data[d])
        display_data.append(np.ones((PCA_dim), dtype=np.float))
    return display_data

def try_pca_by_compounds_with_clustering(pca_dim, compounds, all_data, data_dict):
    PCA_dim = pca_dim
    pca = PCA_torch(center=False, n_components=PCA_dim)
    all_data = np.array(all_data)
    print("number of all data: ", all_data.shape[0])
    _ = pca.fit_PCA(all_data)

    # dis_num = 0
    # for c in compounds:
    #    dis_num += len(data_dict[c])

    display_data = []  # np.zeros((dis_num, 10), np.float)
    # fig, axs = plt.subplots(len(compounds)+1)
    for i in range(len(compounds)):

        comp_data = data_dict[compounds[i]]
        num_data = len(comp_data)
        print("number of data for compound " + compounds[i], num_data)
        comp_data = np.array(comp_data)
        new_comp_data = pca.test(comp_data)
        #clustering = MeanShift(bandwidth=1.5).fit(new_comp_data)
        clustering = KMeans(n_clusters=30, random_state=0).fit(new_comp_data)
        for d in range(clustering.cluster_centers_.shape[0]):
            display_data.append(clustering.cluster_centers_[d, :])
        display_data.append(np.ones((PCA_dim), dtype=np.float))
    return display_data
    # plt.show()

def save_im_pca_by_compounds_with_clustering(pca_dim, all_data, data_dict={}, save_path="./results/pca_ims/"):
    pca = PCA_torch(center=False, n_components=pca_dim)
    all_data = np.array(all_data)
    print("number of all data: ", all_data.shape[0])
    new_data = pca.fit_PCA(all_data)

    min_v = np.min(new_data)
    max_v = np.max(new_data)
    if max_v == min_v:
        exception = "the max value equals to the min value"
        raise exception
    # dis_num = 0
    # for c in compounds:
    #    dis_num += len(data_dict[c])

    padded_data = []  # np.zeros((dis_num, 10), np.float)
    # fig, axs = plt.subplots(len(compounds)+1)
    for c in data_dict.keys():

        comp_data = data_dict[c]
        num_data = len(comp_data)
        print("number of data for compound " + c, num_data)
        comp_data = np.array(comp_data)
        new_comp_data = pca.test(comp_data)
        #clustering = MeanShift(bandwidth=1.5).fit(new_comp_data)

        if new_comp_data.shape[0] > pca_dim:
            clustering = KMeans(n_clusters=pca_dim, random_state=0).fit(new_comp_data)
            im_data = clustering.cluster_centers_
        else:
            padd_num = pca_dim - new_comp_data.shape[0]
            choice = np.random.choice(np.arange(new_comp_data.shape[0]), size=padd_num)
            padded_data = new_comp_data[choice]
            im_data = np.zeros((pca_dim, pca_dim), np.float)
            im_data[:new_comp_data.shape[0], :] = new_comp_data
            im_data[new_comp_data.shape[0]:, :] = padded_data
        im = np.array((im_data - min_v) / (max_v-min_v) * 255, np.uint8)
        cv2.imwrite(save_path+c+".png", im)

if __name__ == '__main__':
    #x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
    #                             label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    #x_eval, y_eval = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_set.csv",
    #                           label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_label.csv")

    x_train = load_cleaned_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv")
    x_eval = load_cleaned_data("/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_set.csv")
    #x_all = np.ones((x_train.shape[0]+x_eval.shape[0], x_train.shape[1]), dtype=x_train.dtype)
    #x_all[:x_train.shape[0], :] = x_train
    #x_all[:x_eval.shape[0], :] = x_eval
    train_feature, eval_feature = try_PCA_with_test(x_train, x_eval)
    #print("number of train data", x_train.shape[0])
    #print("number of eval data", x_eval.shape[0])
    #plot_dist_train_test(train_feature, eval_feature, save_path="./results/pca_no_class/train_eval/")
    #plot_dist_no_label(new_data[::2], save_path="./results/pca_no_class/train/")
    #plot_dist_no_label(new_data[1::2], save_path="./results/pca_no_class/eval/")
    #plot_dist_with_label(new_data[::2], y_train[::2], save_path="./results/pca_with_class/train/")
    #plot_dist_with_label(new_data[1::2], y_train[1::2], save_path="./results/pca_with_class/eval/")
    #plot_dist_name(new_data, save_path="./results/pca_with_name/train/")
    plot_dist_train_test(train_feature, eval_feature, save_path="./results/cleaned_data/train_eval_dist/")
    plot_tsne_train_test(train_feature, eval_feature, save_path="./results/cleaned_data/train_eval_tsne/")