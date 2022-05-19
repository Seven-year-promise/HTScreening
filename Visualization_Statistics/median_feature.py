import matplotlib.pyplot as plt
import csv
import numpy as np
from utils import RAW_CLASSES, load_feature_data_all_by_compound
from scipy.stats import ttest_ind

def t_test_visualize_save(data={}, action={}, save_path=""):
    compound_names = data.keys()
    median_data = []
    wt_data = np.array(data["C0"])
    median_data.append(["C0"] + np.median(wt_data, axis=0).tolist() + ["Wild type"] + [0])
    for comp_name in compound_names:
        if comp_name != "C0":
            comp_data = data[comp_name]
            comp_data = np.array(comp_data)
            median_data.append([comp_name] + np.median(comp_data, axis=0).tolist() + action[comp_name])


    with open(save_path + "median_compounds_feature_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(median_data)



if __name__ == "__main__":
    feature_data_by_compound, action_dict_by_compound = load_feature_data_all_by_compound(path="/srv/yanke/PycharmProjects/HTScreening/data/featured/all_compounds_feature_fish_with_action.csv")
    t_test_visualize_save(feature_data_by_compound, action_dict_by_compound, save_path="/srv/yanke/PycharmProjects/HTScreening/data/median/")
