import numpy as np
import nthresh
import csv
import sys
sys.path.append("../")
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

"""
visualize the intergrated feature
"""


def draw_box_plot_ttest(all_dis, dis_save):
    WILD_dis = all_dis[0]

    x_labels = []
    fig, ax = plt.subplots()
    for n, (a_d, d) in enumerate(zip(all_dis, dis_save)):
        if n == 0:
            print(WILD_dis, a_d)
        F, p = ttest_ind(WILD_dis, a_d, equal_var=False)
        plt.text(n+1, d[1] + d[4] + 0.5, str(round(p, 3)), fontsize=8)
        x_labels.append(d[0])
    ax.boxplot(all_dis, labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=85, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.show()

def visualize_separate_wt(data):
    fig, ax = plt.subplots()
    box_data = []
    box_labels = []
    for l in [-1,20+11,33+11]: #range(label_num+1):
        # display intergration
        if l < 0:
            for w_i in range(12):
                the_data = data["C"+str(w_i)]

                if len(the_data) < 1:
                    continue
                the_data = np.array(the_data).reshape(len(the_data),)
                print(the_data)
                box_data.append(the_data)
                box_labels.append("C0_"+str(w_i))
        else:
            the_data = data["C" + str(l)]
            if len(the_data) < 1:
                continue
            the_data = np.array(the_data).reshape(len(the_data), )
            box_data.append(the_data)
            box_labels.append("C" + str(l-11))
    print(box_data)
    ax.boxplot(box_data, labels=box_labels)
    plt.ylabel("the integration of the time series")
    plt.setp(ax.get_xticklabels(), rotation=85, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.show()

def visualize_together_wt(data, save_path):
    no_effected_comp = []
    no_effected_comp.append("no_effects")
    higher_effected_comp = []
    higher_effected_comp.append("higher_effects")
    lower_effected_comp =[]
    lower_effected_comp.append("lower_effects")
    data_num = len(data.keys())
    fig, ax = plt.subplots()
    box_data = []
    box_labels = []
    for l in [-1] + list(range(12, data_num+1)):#,20+11,33+11]: #range(label_num+1):
        # display intergration
        if l < 0:
            wild_data = []
            for w_i in range(12):
                the_data = data["C"+str(w_i)]
                print(len(the_data))
                if len(the_data) < 1:
                    continue
                the_data = np.array(the_data).reshape(len(the_data),)
                #print(the_data)
                wild_data += the_data.tolist()
                print(len(wild_data))
            box_data.append(wild_data)
            box_labels.append("C0")
        else:
            if "C" + str(l) in data.keys():
                the_data = data["C" + str(l)]
                if len(the_data) < 1:
                    continue
                the_data = np.array(the_data).reshape(len(the_data), )
                box_data.append(the_data)
                box_labels.append("C" + str(l-11))
    #print(box_data)

    WILD_inte = box_data[0]

    for n, (b_d, b_l) in enumerate(zip(box_data, box_labels)):
        F, p = ttest_ind(WILD_inte, b_d, equal_var=False)
        if p < 0.05:
            if np.mean(WILD_inte) > np.mean(b_d):
                lower_effected_comp.append(b_l)
            else:
                higher_effected_comp.append(b_l)
        else:
            no_effected_comp.append(b_l)
        plt.text(n + 1, np.max(b_d), str(round(p, 3)), fontsize=8)

    ax.boxplot(box_data, labels=box_labels)
    plt.ylabel("the integration of the time series")
    plt.setp(ax.get_xticklabels(), rotation=85, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.show()
    with open(save_path + "effects_or_not_with_integration.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerow(no_effected_comp)
        csv_writer.writerow(higher_effected_comp)
        csv_writer.writerow(lower_effected_comp)

if __name__ == "__main__":
    feature_data_by_compound, action_dict_by_compound = load_feature_data_all_by_compound(
        path="/Users/yankeewann/Desktop/HTScreening/data/featured/all_compounds_integration_feature_fish_with_action_wt_separate.csv")

    #visualize_separate_wt(feature_data_by_compound)
    visualize_together_wt(feature_data_by_compound, "/Users/yankeewann/Desktop/HTScreening/data/")
    #mean_distance_with_PCA_visualize(data, labels)
