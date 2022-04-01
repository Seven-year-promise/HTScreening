import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from Analysis import f_one_way_test, MANOVA_test, outlier_remove, feature_normalize, f_one_way_test_selected, t_test_selected, MANOVA_test_selected
from utils import CLASSES, load_effected_data_feature, exist_key, get_key


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
    data = feature_normalize(data)
    data, actions = outlier_remove(data, actions)
    print(data.shape, actions.shape)
    F, p = MANOVA_test(data, actions)
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
        axs[a].set_ylim(0, 0.05)
        axs[a].margins(x=0)
        #plt.grid(b=True, which="both", axis="both")
        # plt.ylabel(ylabels[2], fontsize=8)
        # plt.title(titles[2])
        #plt.xticks(fontsize=14, fontname="Times New Roman")
        #plt.yticks(fontsize=14, fontname="Times New Roman")
        #plt.title(ylabels[2], fontname="Times New Roman", fontsize=14)
    for i, (f_v, p_v) in enumerate(zip(F, p)):
        axs[0].text(i, 0.065, s="F:" + str(round(f_v, 6)), color="tab:red")
        axs[0].text(i, 0.060, s="p:" + str(round(p_v, 10)), color="tab:red")
    plt.tight_layout()
    plt.show()
    #fig.savefig(save_path + "median14feature_action_box_plot_scale-with_anova.png")

def visualize_action_binary_anova_box_plot(data, actions, num_actions, save_path):
    data = feature_normalize(data)
    data, actions = outlier_remove(data, actions)
    print(data.shape, actions.shape)

    wild_inds = actions == 0
    wild_data = data[wild_inds, :]
    for a in range(1, num_actions):

        action_inds = actions==a
        action_data = data[action_inds, :]
        F, p = f_one_way_test_selected(data, actions, a)
        print(np.sum(action_inds*1))
        #axs[a].set_size_inches(30, 10)

        fig, axs = plt.subplots(2, figsize=(30, 10))

        box_data = []
        labels = []
        for a_d in range(wild_data.shape[1]):
            box_data.append(wild_data[:, a_d])
            labels.append(a_d)
        axs[0].boxplot(box_data, labels=labels, widths=0.5, positions=range(wild_data.shape[1]), patch_artist=True,
                       showfliers=True, showmeans=True)
        axs[0].set_ylabel("14 feature")
        axs[0].set_xlabel("Feature index")
        axs[0].set_title(get_key(CLASSES, 0))
        axs[0].set_ylim(0, 0.05)
        axs[0].margins(x=0)

        box_data = []
        labels = []
        for a_d in range(action_data.shape[1]):
            box_data.append(action_data[:, a_d])
            labels.append(a_d)
        axs[1].boxplot(box_data, labels=labels, widths=0.5, positions=range(action_data.shape[1]), patch_artist=True,
                             showfliers=True, showmeans=True)
        axs[1].set_ylabel("14 feature")
        axs[1].set_xlabel("Feature index")
        axs[1].set_title(get_key(CLASSES, a))
        axs[1].set_ylim(0, 0.05)
        axs[1].margins(x=0)
        #plt.grid(b=True, which="both", axis="both")
        # plt.ylabel(ylabels[2], fontsize=8)
        # plt.title(titles[2])
        #plt.xticks(fontsize=14, fontname="Times New Roman")
        #plt.yticks(fontsize=14, fontname="Times New Roman")
        #plt.title(ylabels[2], fontname="Times New Roman", fontsize=14)
        for i, (f_v, p_v) in enumerate(zip(F, p)):
            axs[0].text(i, 0.055, s="F:" + str(round(f_v, 5)), color="tab:red", size=14)
            axs[0].text(i, 0.050, s="p:" + str(round(p_v, 8)), color="tab:red", size=14)
        plt.tight_layout()
        #plt.show()
        fig.savefig(save_path + "anova_wild_vs_action_" + get_key(CLASSES, a) + ".png")
        plt.clf()

def visualize_action_binary_ttest_box_plot(data, actions, num_actions, save_path):
    data = feature_normalize(data)
    data, actions = outlier_remove(data, actions)
    print(data.shape, actions.shape)

    wild_inds = actions == 0
    wild_data = data[wild_inds, :]
    for a in range(1, num_actions):

        action_inds = actions==a
        action_data = data[action_inds, :]
        F, p = t_test_selected(data, actions, a)
        print(np.sum(action_inds*1))
        #axs[a].set_size_inches(30, 10)

        fig, axs = plt.subplots(2, figsize=(30, 10))

        box_data = []
        labels = []
        for a_d in range(wild_data.shape[1]):
            box_data.append(wild_data[:, a_d])
            labels.append(a_d)
        axs[0].boxplot(box_data, labels=labels, widths=0.5, positions=range(wild_data.shape[1]), patch_artist=True,
                       showfliers=True, showmeans=True)
        axs[0].set_ylabel("14 feature")
        axs[0].set_xlabel("Feature index")
        axs[0].set_title(get_key(CLASSES, 0))
        axs[0].set_ylim(0, 0.05)
        axs[0].margins(x=0)

        box_data = []
        labels = []
        for a_d in range(action_data.shape[1]):
            box_data.append(action_data[:, a_d])
            labels.append(a_d)
        axs[1].boxplot(box_data, labels=labels, widths=0.5, positions=range(action_data.shape[1]), patch_artist=True,
                             showfliers=True, showmeans=True)
        axs[1].set_ylabel("14 feature")
        axs[1].set_xlabel("Feature index")
        axs[1].set_title(get_key(CLASSES, a))
        axs[1].set_ylim(0, 0.05)
        axs[1].margins(x=0)
        #plt.grid(b=True, which="both", axis="both")
        # plt.ylabel(ylabels[2], fontsize=8)
        # plt.title(titles[2])
        #plt.xticks(fontsize=14, fontname="Times New Roman")
        #plt.yticks(fontsize=14, fontname="Times New Roman")
        #plt.title(ylabels[2], fontname="Times New Roman", fontsize=14)
        for i, (f_v, p_v) in enumerate(zip(F, p)):
            axs[0].text(i, 0.055, s="F:" + str(round(f_v, 5)), color="tab:red", size=14)
            axs[0].text(i, 0.050, s="p:" + str(round(p_v, 8)), color="tab:red", size=14)
        plt.tight_layout()
        #plt.show()
        fig.savefig(save_path + "ttest_wild_vs_action_" + get_key(CLASSES, a) + ".png")
        plt.clf()

def visualize_action_binary_manova_box_plot(data, actions, num_actions):
    data = feature_normalize(data)
    data, actions = outlier_remove(data, actions)
    print(data.shape, actions.shape)

    wild_inds = actions == 0
    wild_data = data[wild_inds, :]
    for a in range(1, num_actions):
        print("wild VS actions: ", get_key(CLASSES, a))
        MANOVA_test_selected(data, actions, a)

if __name__ == "__main__":
    data, comps, actions = load_effected_data_feature(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_fishes_labeled.csv")
    #visualize_action_binary_ttest_box_plot(data, actions, 4, save_path="./effected_feature_fish/binary_statistic/")
    visualize_action_binary_manova_box_plot(data, actions, 4)
