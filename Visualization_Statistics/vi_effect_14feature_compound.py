import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from Analysis import f_one_way_test
from utils import CLASSES, load_effected_data_feature_median, exist_key, get_key


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
    F, p = f_one_way_test(data, actions)
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
    for i, (f_v, p_v) in enumerate(zip(F, p)):
        axs[0].text(i, 0.108, s="F:" + str(round(f_v, 4)), color="tab:red")
        axs[0].text(i, 0.101, s="p:" + str(round(p_v, 4)), color="tab:red")
    plt.tight_layout()
    #plt.show()
    fig.savefig(save_path + "median14feature_action_box_plot_scale-with_anova.png")

if __name__ == "__main__":
    data, comps, actions = load_effected_data_feature_median(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    visualize_action_box_plot(data, actions, 4, save_path="./feature_median_14/")
