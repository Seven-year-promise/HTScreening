import numpy as np
import csv
import sys
sys.path.append("./Methods/PCA/")
from pca_dim_reduce import PCA_torch
"""
extract feature based on PCA
"""

def extract_feature_to(data_path, save_path):

    cleaned_data = []
    cleaned_comp_names = []
    cleaned_action_infos = []
    all_data_feature = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            #print(one_data)
            one_data = [float(i) for i in l[1:-2]]
            #print(one_data)
            if len(one_data) != 539:
                print("oops!")
                continue
            cleaned_data.append(one_data)
            cleaned_comp_names.append(l[0])
            cleaned_action_infos.append(l[-2:])

    cleaned_data = np.array(cleaned_data)
    pca = PCA_torch(center=False, n_components=5)
    new_cleaned_data = pca.fit_PCA(cleaned_data)

    for i in range(new_cleaned_data.shape[0]):
        feature = new_cleaned_data[i, :]
        feature_data = [cleaned_comp_names[i]] + feature.tolist() + cleaned_action_infos[i]
        all_data_feature.append(feature_data)

    with open(save_path + "all_compounds_pca5_feature_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_data_feature)

if __name__ == "__main__":
    data_path = "./data/cleaned/all_compounds_ori_fish_with_action.csv"
    save_path = "./data/featured/"
    extract_feature_to(data_path, save_path)