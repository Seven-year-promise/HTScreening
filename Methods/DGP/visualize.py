import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer

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

def visualize_action(data, actions, actions_to_show):
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
    data, comps, actions = load_filtered_data(path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/saved_data/2d/filtered_comp_mu_action.csv")
    visualize_action(data, actions, ["Wildtype", "TRPV agonist", "GABAA allosteric antagonist", "GABAA pore blocker"])
    #visualize_compound(data, actions, comps, 130, save_path="./visualization/by_compounds/filtered/")