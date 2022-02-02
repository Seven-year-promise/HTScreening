from pca_dim_reduce import load_data, try_PCA_with_torch
import csv
import numpy as np
import matplotlib.pyplot as plt

def read_pattern_index(path):
    pattern_inds = []
    with open(path, newline='') as ind_f:
        read_lines = ind_f.readlines()
        for j, l in enumerate(read_lines):
            d_l = l.split(";")
            print(d_l)
            data_line = [int(i) for i in d_l[1:-1]]
            pattern_inds.append(data_line)

    return pattern_inds


if __name__ == '__main__':
    x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
                                 label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    pattern_indexes = read_pattern_index(path="/srv/yanke/PycharmProjects/HTScreening/Methods/PCA/results/pca_with_name/train/pattern-slected.txt")
    _, new_data = try_PCA_with_torch(x_train)
    print("number of data", x_train.shape[0])
    #fig = plt.figure()
    #color = plt.cm.Set1(0)
    """
    #draw the data after pca
    for p in pattern_indexes:
        for p_i in p:
            plt.scatter(new_data[p_i, 0], new_data[p_i, 1], s=1, color=color)  # , label="all_data")
            plt.text(new_data[p_i, 0], new_data[p_i, 1], p_i, fontsize=6)  # , color=color, label="all_data")
    """

    #fig, axs = plt.subplots(len(pattern_indexes))

    # ==== draw all data of all patterns =======
    """
    for i, p in enumerate(pattern_indexes):
        for p_i in p:
            axs[i].plot(x_train[p_i, :])
        axs[i].set_ylabel("motion index")
        axs[i].set_xlabel("time")
        #axs[i].set_title("pattern " + str(i+1))
    """
    # ==== draw the average of all patterns =======
    fig = plt.figure()
    for i, p in enumerate(pattern_indexes):
        patten_data = []
        for p_i in p:
            patten_data.append(x_train[p_i, :])
        patten_data = np.average(patten_data, axis=0)
        plt.plot(patten_data, label = "pattern " + str(i))
    plt.ylabel("motion index")
    plt.xlabel("time")


    # ==== draw all data with index for pattern 1 ======
    """
    fig = plt.figure()
    for p_i in pattern_indexes[0]:
        plt.plot(x_train[p_i, :], label=str(p_i))
    plt.ylabel("motion index")
    plt.xlabel("time")
    """
    # axs[i].set_title("pattern " + str(i+1))
    #plt.title("Select data for each pattern")
    plt.legend(loc="best")
    # fig.savefig(save_path  + "VAE_embedding_" + str(ic) + ".png")
    plt.show()
    # fig.savefig(save_path +  "VAE_embedding.png")
    #pickle.dump(fig, open(save_path + "VAE_embedding.pickle", "wb"))
    #plt.clf()