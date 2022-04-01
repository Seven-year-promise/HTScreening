import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from data_loader import load_filtered_effected_data

RAW_CLASSES = {"Wildtype": 0,
               "GABAA pore blocker": 1,
               "vesicular ACh transport antagonist": 2,
               "nAChR orthosteric agonist": 3,
               "nAChR orthosteric antagonist": 4,
               "TRPV agonist": 5,
               "GABAA allosteric antagonist": 6,
               "RyR agonist": 7,
               "Na channel": 8,
               "unknown": 9
               }
CLASSES = {"WT_control": 0,
           "TRPV agonist": 1,
           "GABAA allosteric antagonist": 2,
           "GABAA pore blocker": 3}

def load_filtered_data(path):
    compounds = []
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j > 0:
                comp_name = l[0]
                #print(comp_name)
                if comp_name[0] == "W":
                    compound = 0
                else:
                   compound = int(comp_name[1:])
                compounds.append(compound)
                data_line = [float(i) for i in l[1:-1]]
                data_list.append(data_line)
                data_labels.append(int(l[-1]))

    print("number of data", len(data_labels))
    print("dimension of data", len(data_list[0]))
    #print(compounds)
    return np.array(data_list), np.array(compounds), np.array(data_labels)

def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k
    return "None"

def visualize_selected_action(data, actions, actions_to_show):
    fig, axs = plt.subplots(len(actions_to_show))
    for a, a_s in enumerate(actions_to_show):
        a_ind = RAW_CLASSES[a_s]
        action_inds = actions==a_ind
        action_data = data[action_inds, :]
        print(action_inds)
        for a_d in range(action_data.shape[0]):
            axs[a].plot(action_data[a_d])
        axs[a].set_ylabel("motion index")
        axs[a].set_xlabel("time")
        axs[a].set_title(a_s)
        axs[a].margins(x=0)
    plt.tight_layout()
    plt.show()

def visualize_by_action(data, actions):
    num_actions = np.max(actions)+1
    fig, axs = plt.subplots(num_actions)
    for a in range(num_actions):
        action_inds = actions==a
        action_data = data[action_inds, :]
        print(action_inds)
        for a_d in range(action_data.shape[0]):
            axs[a].plot(action_data[a_d])
        axs[a].set_ylabel("motion index")
        axs[a].set_xlabel("time")
        axs[a].set_title(get_key(CLASSES, a))
        axs[a].margins(x=0)
    plt.tight_layout()
    plt.show()

def visualize_mu_by_action(data, actions, save_path):
    num_actions = np.max(actions)+1
    #fig, axs = plt.subplots(num_actions)
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    color2 = plt.cm.Set1(2)
    color3 = plt.cm.Set1(3)
    colors = [color0, color1, color2, color3]
    for a in range(num_actions):
        action_inds = actions==a
        action_data = data[action_inds, :]
        print(action_inds)
        #axs[a].scatter(data[:, 0], data[:, 1], s=5, color=color0)
        plt.scatter(action_data[:, 0], action_data[:, 1], s=5, color=colors[a])
        #axs[a].set_ylabel("motion index")
        #axs[a].set_xlabel("time")
        #axs[a].set_title(get_key(CLASSES, a))
        #axs[a].margins(x=0)
    #plt.xlim(-5, 1)
    #plt.ylim(-4, 4)
    plt.tight_layout()
    plt.savefig(save_path + "ori_mu2.png")

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
        plt.title(get_key(RAW_CLASSES, action_ind))
        plt.margins(x=0)
        plt.tight_layout()
        fig.savefig(save_path + "C" + str(c) + ".png")

if __name__ == "__main__":
    #data, comps, actions = load_filtered_data(path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/saved_data/2d/filtered_comp_data_action.csv")
    data, actions = load_filtered_effected_data(
        path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/original_comp_mu_action2.csv", normalize=False)
    visualize_mu_by_action(data, actions.reshape(-1, ), save_path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/visualization/by_actions/effected/")
    #visualize_action(data, actions.reshape(-1, ), ["Wildtype", "TRPV agonist", "GABAA allosteric antagonist", "GABAA pore blocker"])
    #visualize_compound(data, actions, comps, 130, save_path="./visualization/by_compounds/filtered/")