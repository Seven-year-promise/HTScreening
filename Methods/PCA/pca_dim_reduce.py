import sys
#sys.path.insert(0, "/srv/yanke/PycharmProjects/HTScreening")
import numpy as np

import sys
sys.path.append("../../")
import csv
from Methods.PCA.PCA import PCA_torch
from Methods.PCA.plot_tsne import plot_dist_no_label, plot_dist_with_label, plot_dist_name, plot_dist_train_test

def load_data(data_path, label_path):

    data_list = []
    data_labels = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)

    with open(label_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_labels = [int(i) for i in l]

    x = np.array(data_list)
    y = np.array(data_labels)#.reshape(-1, 1)
    return x, y

def try_PCA_with_torch(data):
    pca = PCA_torch(center=False, n_components=2)
    new_feature = pca.fit_PCA(data)

    return data, new_feature

def try_PCA__with_test(train_set, test_set):
    pca = PCA_torch(center=False, n_components=2)
    new_train = pca.fit_PCA(train_set)
    new_test = pca.test(test_set)

    return new_train, new_test

if __name__ == '__main__':
    x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
                                 label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    x_eval, y_eval = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_set.csv",
                               label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/eval_label.csv")

    #x_all = np.ones((x_train.shape[0]+x_eval.shape[0], x_train.shape[1]), dtype=x_train.dtype)
    #x_all[:x_train.shape[0], :] = x_train
    #x_all[:x_eval.shape[0], :] = x_eval
    train_feature, eval_feature = try_PCA__with_test(x_train, x_eval)
    print("number of train data", x_train.shape[0])
    print("number of eval data", x_eval.shape[0])
    plot_dist_train_test(train_feature, eval_feature, save_path="./results/pca_no_class/train_eval/")
    #plot_dist_no_label(new_data[::2], save_path="./results/pca_no_class/train/")
    #plot_dist_no_label(new_data[1::2], save_path="./results/pca_no_class/eval/")
    #plot_dist_with_label(new_data[::2], y_train[::2], save_path="./results/pca_with_class/train/")
    #plot_dist_with_label(new_data[1::2], y_train[1::2], save_path="./results/pca_with_class/eval/")
    #plot_dist_name(new_data, save_path="./results/pca_with_name/train/")