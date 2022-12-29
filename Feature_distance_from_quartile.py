import numpy as np
import nthresh
import csv
import sys
sys.path.append("/")
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from config import *

"""
0: no effected
1: lower effected
2: higher effected
get the binary_code feature with actions
000:001:002:010:011:012:020:021:022:
100:101:102:110:111:112:120:121:122:
200:201:202:210:211:212:220:221:222:
"""
p_thre = 0.05
feature_num = 14
control_name = "all"

def get_binary_code_action(inds):
    code_v = 0
    for i, x in enumerate(inds):
        code_v += x * 3 ** (feature_num - 1 - i)
    return code_v

def visualize_together_wt(data, action_dict, save_path):
    binary_code_features = []
    binary_code_features.append(["Compound"] + ["Feature " + str(x) for x in range(feature_num)] + ["Action", "Pattern"])
    data_num = len(data.keys())
    fig, ax = plt.subplots()
    box_data = []
    box_labels = []
    for k_i in range(180):#data.keys():
        k="C"+str(k_i)
        if k in data.keys():
            the_data = data[k]
            if len(the_data) < 1:
                continue
            the_data = np.array(the_data).reshape(len(the_data), -1)
            if k == "C0":
                WILD_inte = the_data
            box_data.append(the_data)
            box_labels.append(k)

    #print(box_data)
    #print(box_data)
    #WILD_inte = box_data[0]
    WILD_inte = np.array(WILD_inte)
    #print(WILD_inte)

    for n, (b_d, b_l) in enumerate(zip(box_data, box_labels)):
        b_d = np.array(b_d)
        binary_code_data = [b_l]

        inds = []
        for ph in range(feature_num):
            F, p = ttest_ind(WILD_inte[:, ph], b_d[:,ph], equal_var=False)
            if p < p_thre:
                if np.mean(WILD_inte[:, ph]) > np.mean(b_d[:,ph]):
                    inds.append(1)
                    binary_code_data.append(1)
                else:
                    inds.append(2)
                    binary_code_data.append(2)
            else:
                inds.append(0)
                binary_code_data.append(0)

        code_str = ""
        for x in inds:
            code_str += str(x)
        #binary_code_data.append(code_str)
        binary_code_data.append(action_dict[b_l][0])
        binary_code_data.append(get_binary_code_action(inds))
        binary_code_features.append(binary_code_data)

    with open(save_path / ("effects_binary_codes_with_"+str(feature_num)+"quartile" + str(p_thre)+control_name+".csv"), "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(binary_code_features)

if __name__ == "__main__":
    feature_data_by_compound, action_dict_by_compound = load_feature_data_all_by_compound(
        path=SAVE_FEATURE_PATH / ("all_compounds_" +str(feature_num)+"quartile_feature_fish_with_action_wt_"+control_name+".csv"))

    #visualize_separate_wt(feature_data_by_compound)
    print(feature_data_by_compound.keys())
    visualize_together_wt(feature_data_by_compound, action_dict_by_compound, SAVE_FEATURE_PATH)
    #mean_distance_with_PCA_visualize(data, labels)
