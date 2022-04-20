import numpy as np
import csv
"""
extract feature based on
[0:187) before the first light
[187:363) after the first light
[363:) after the second light
"""
PHASES = [0, 187, 207, 227, 267, 307, 363, 383, 403, 443, 483, -1]

def feature_extract(data):
    features = []
    for i in range(len(PHASES)-1):
        phase_data = data[PHASES[i]:PHASES[i+1]]
        l_q = np.quantile(phase_data, 0.25)
        h_q = np.quantile(phase_data, 0.75)
        features.append(l_q)
        features.append(h_q)

    return features

def extract_feature_to(data_path, save_path):
    all_data_feature = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            one_data = [float(i) for i in l[1:-2]]
            feature = feature_extract(one_data)
            feature_data = [l[0]] + feature + l[-2:]
            all_data_feature.append(feature_data)
    with open(save_path + "all_compounds_feature_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_data_feature)

if __name__ == "__main__":
    data_path = "./data/cleaned/all_compounds_ori_fish_with_action.csv"
    save_path = "./data/featured/"
    extract_feature_to(data_path, save_path)