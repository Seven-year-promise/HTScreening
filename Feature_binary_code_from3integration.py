import numpy as np
import nthresh
import csv
import sys
sys.path.append("/")
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

"""
0: no effected
1: lower effected
2: higher effected
get the binary_code feature with actions
000:001:002:010:011:012:020:021:022:
100:101:102:110:111:112:120:121:122:
200:201:202:210:211:212:220:221:222:
"""

def get_binary_code_action(ind0, ind1, ind2):
    return ind0*9 + ind1*3 + ind2

def visualize_together_wt(data, action_dict, save_path):
    binary_code_features = []
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
                the_data = np.array(the_data).reshape(len(the_data),-1)
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
                the_data = np.array(the_data).reshape(len(the_data), -1)
                box_data.append(the_data)
                box_labels.append("C" + str(l-11))
    #print(box_data)
    #print(box_data)
    WILD_inte = box_data[0]
    WILD_inte = np.array(WILD_inte)
    #print(WILD_inte)

    for n, (b_d, b_l) in enumerate(zip(box_data, box_labels)):
        b_d = np.array(b_d)
        binary_code_data = [b_l]
        # for phase 0
        F, p = ttest_ind(WILD_inte[:, 0], b_d[:,0], equal_var=False)
        if p < 0.05:
            if np.mean(WILD_inte) > np.mean(b_d):
                ind0 = 1
                binary_code_data.append(1)
            else:
                ind0 = 2
                binary_code_data.append(2)
        else:
            ind0 = 0
            binary_code_data.append(0)

        # for phase 1
        F, p = ttest_ind(WILD_inte[:, 1], b_d[:, 1], equal_var=False)
        if p < 0.05:
            if np.mean(WILD_inte) > np.mean(b_d):
                ind1 = 1
                binary_code_data.append(1)
            else:
                ind1 = 2
                binary_code_data.append(2)
        else:
            ind1 = 0
            binary_code_data.append(0)

        # for phase 2
        F, p = ttest_ind(WILD_inte[:, 2], b_d[:, 2], equal_var=False)
        if p < 0.05:
            if np.mean(WILD_inte) > np.mean(b_d):
                ind2 = 1
                binary_code_data.append(1)
            else:
                ind2 = 2
                binary_code_data.append(2)
        else:
            ind2 = 0
            binary_code_data.append(0)

        binary_code_data.append(str(ind0)+str(ind1)+str(ind2))
        binary_code_data.append(get_binary_code_action(ind0, ind1, ind2))
        binary_code_features.append(binary_code_data)

    with open(save_path + "effects_binary_codes_with_integration.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(binary_code_features)

if __name__ == "__main__":
    feature_data_by_compound, action_dict_by_compound = load_feature_data_all_by_compound(
        path="/data/featured/all_compounds_3integration_feature_fish_with_action_wt_separate.csv")

    #visualize_separate_wt(feature_data_by_compound)
    visualize_together_wt(feature_data_by_compound, action_dict_by_compound, "/data/featured/")
    #mean_distance_with_PCA_visualize(data, labels)
