import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer

CLASSES = {"Wildtype": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

def load_effected_data(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                action_name = l[-2]
                if exist_key(CLASSES, action_name):
                    comp_name = l[0]
                    #print(comp_name)
                    if comp_name[0] == "s":
                        compound = 0
                    else:
                       compound = int(comp_name[1:])
                    compounds.append(compound)
                    data_line = [float(i) for i in l[1:-2]]
                    data_list.append(data_line)
                    data_labels.append(CLASSES[l[-2]])

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)

def exist_key(dict, key):
    for k, v in dict.items():
        if k == key:
            return True

    return False

def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return "None"

def visualize_action(data, actions, num_actions, save_path):
    fig, axs = plt.subplots(num_actions, figsize=(30, 10))
    for a in range(num_actions):
        action_inds = actions==a
        action_data = data[action_inds, :]
        print(action_inds)
        #axs[a].set_size_inches(30, 10)
        for a_d in range(action_data.shape[0]):
            axs[a].plot(action_data[a_d])
        axs[a].set_ylabel("14 feature after median")
        axs[a].set_xlabel("Feature index")
        axs[a].set_title(get_key(CLASSES, a))
        axs[a].set_ylim(0, 0.4)
        axs[a].margins(x=0)
    plt.tight_layout()
    #plt.show()
    fig.savefig(save_path + "median14feature_action.png")

def visualize_action_box_plot(data, actions, num_actions, save_path):
    fig, axs = plt.subplots(num_actions, figsize=(30, 10))
    for a in range(num_actions):
        action_inds = actions==a
        action_data = data[action_inds, :]
        print(np.sum(action_inds*1))
        #axs[a].set_size_inches(30, 10)
        box_data = []
        labels = []
        for a_d in range(action_data.shape[1]):
            box_data.append(action_data[:, a_d])
            labels.append(a_d)
        axs[a].boxplot(box_data, labels=labels, widths=0.5, positions=range(action_data.shape[1]), patch_artist=True,
                             showfliers=True, showmeans=True)
        axs[a].set_ylabel("14 feature after median")
        axs[a].set_xlabel("Feature index")
        axs[a].set_title(get_key(CLASSES, a))
        axs[a].set_ylim(0, 0.1)
        axs[a].margins(x=0)
        #plt.grid(b=True, which="both", axis="both")
        # plt.ylabel(ylabels[2], fontsize=8)
        # plt.title(titles[2])
        #plt.xticks(fontsize=14, fontname="Times New Roman")
        #plt.yticks(fontsize=14, fontname="Times New Roman")
        #plt.title(ylabels[2], fontname="Times New Roman", fontsize=14)

    plt.tight_layout()
    #plt.show()
    fig.savefig(save_path + "median14feature_action_box_plot_scale.png")

if __name__ == "__main__":
    data, comps, actions = load_effected_data(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    visualize_action_box_plot(data, actions, 4, save_path="./feature_median_14/")
