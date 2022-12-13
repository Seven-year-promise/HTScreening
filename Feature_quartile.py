import numpy as np
import csv
from config import *

"""
extract feature based on
[0:187) before the first light
[187:363) after the first light
[363:) after the second light
"""
"""
two features : 
1) Qantile 3 
2) Quantile 1
"""
#PHASES = [0, 187, 207, 227, 267, 307, 363, 383, 403, 443, 483, -1] # 11 phases

PHASES = [0, 187, 207, 235, 301, 363, 408, -1]
control_name= "all"

def feature_extract(data):
    features = []
    for i in range(len(PHASES)-1):
        phase_data = data[PHASES[i]:PHASES[i+1]]
        l_q = np.quantile(phase_data, 0.25)
        h_q = np.quantile(phase_data, 0.75)
        #median_v = np.median(phase_data)
        features.append(l_q)
        features.append(h_q)

    return features

def extract_feature_to(data_path):
    all_data_feature = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            one_data = [float(i) for i in l[1:-2]]
            feature = feature_extract(one_data)
            feature_data = [l[0]] + feature + l[-2:]
            all_data_feature.append(feature_data)
    with open(SAVE_FEATURE_PATH / ("all_compounds_" +str(len(PHASES)*2-2)+"quartile_feature_fish_with_action_wt_"+control_name+".csv"), "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerow(
            ["Compound", "Pre: Q1", "Pre: Q3", "L1: Q1", "L1: Q3", "E1: Q1", "E1: Q3", "E2: Q1", "E2: Q3", "E3: Q1",
             "E3: Q3", "R1: Q1", "R1: Q3", "R2: Q1", "R2: Q3", "action_name", "action_id"])
        csv_writer.writerows(all_data_feature)

if __name__ == "__main__":
    data_path = SAVE_CLEAN_PATH / ("all_compounds_ori_fish_with_action_wt_"+control_name+".csv")
    extract_feature_to(data_path)