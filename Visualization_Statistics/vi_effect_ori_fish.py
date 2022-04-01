import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from utils import CLASSES_alias, get_key, load_effected_data_ori



def visualize_action(data, actions, num_actions):
    fig, axs = plt.subplots(num_actions)
    for a in range(num_actions):
        action_inds = actions==a
        action_data = data[action_inds, :]
        print(action_inds)
        for a_d in range(action_data.shape[0]):
            axs[a].plot(action_data[a_d])
        axs[a].set_ylabel("motion index")
        axs[a].set_xlabel("time")
        axs[a].set_title(get_key(CLASSES_alias, a))
        axs[a].margins(x=0)
    plt.tight_layout()
    plt.show()

def visualize_compound(data, actions, compounds, num_compounds, save_path=None):
    for c in range(num_compounds):
        compounds_inds = compounds==c
        compounds_data = data[compounds_inds, :]
        if compounds_data.shape[0] <1:
            continue
        action_ind = actions[compounds_inds][0]
        fig = plt.figure()
        fig.set_size_inches(30, 10)
        for a_d in range(compounds_data.shape[0]):
            plt.plot(compounds_data[a_d])
        plt.ylabel("motion index")
        plt.xlabel("time")
        plt.title(get_key(CLASSES_alias, action_ind))
        plt.margins(x=0)
        plt.tight_layout()
        fig.savefig(save_path + "C" + str(c) + ".png")

if __name__ == "__main__":
    data, comps, actions = load_effected_data_ori(path="/srv/yanke/PycharmProjects/HTScreening/data/raw_data/effected_compounds_pvalue_frames_labeled.csv")
    #visualize_action(data, actions, 4)
    visualize_compound(data, actions, comps, 130, save_path="./effect_ori_fish/by_compounds/")