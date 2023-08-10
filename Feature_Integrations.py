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
one feature for each phase : 
integration
"""
#PHASES = [0, 187, 363, -1]

PHASES = [0, 187, 207, 235, 301, 363, 408, -1]
control_name = "all"

def intergration(data):
    return np.sum(data)

def feature_extract(data):
    features = []
    for i in range(len(PHASES)-1):
        phase_data = data[PHASES[i]:PHASES[i+1]]
        feature = intergration(phase_data)
        features.append(feature)

    return features

def extract_feature_to(data_path):

    #cleaned_data = []
    cleaned_feature_data = []
    cleaned_comp_names = []
    cleaned_action_infos = []
    all_data_feature = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            #print(one_data)
            one_data = [float(i) for i in l[1:-2]]
            #print(one_data)
            #if len(one_data) != 539:
            #    print("oops!")
            #    continue
            cleaned_feature_data.append(feature_extract(one_data))
            cleaned_comp_names.append(l[0])
            cleaned_action_infos.append(l[-2:])

    #cleaned_feature_data = np.array(cleaned_feature_data)
    for i in range(len(cleaned_feature_data)):
        feature = cleaned_feature_data[i]
        feature_data = [cleaned_comp_names[i]] + feature + cleaned_action_infos[i]
        all_data_feature.append(feature_data)

    with open(TEST_SAVE_FEATURE_PATH / ("all_compounds_"+str(len(PHASES)-1)+"integration_feature_fish_with_action_wt_"+control_name+".csv"), "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerow(["Feature " + str(x) for x in range(1, len(PHASES))] + ["action_name", "action_id"])
        csv_writer.writerows(all_data_feature)

if __name__ == "__main__":
    data_path = TEST_SAVE_CLEAN_PATH / ("all_compounds_ori_fish_with_action_wt_"+control_name+".csv")
    extract_feature_to(data_path)