import os
import shutil
import csv
import numpy as np
from config import *
import pingouin as pg
import pandas as pd

"""
file_formate:

compound fish index -- data -- action name -- action id

GABAA pore blocker: 1, [2:30)
vesicular ACh transport antagonist: 30 [31,43)
nAChR orthosteric agonist: 43 [44, 52)
nAChR orthosteric antagonist: 52 [53, 63)
TRPV agonist: 63 [64, 122)
GABAA allosteric antagonist: 122 [123, 144)
RyR agonist: 144 [145, 147)
Na channel: 147 [148, 150)
complex II inhibitor: 150 [151, 156)
nAChR allosteric agonist: 156 [157, 159)
unknown-likely neurotoxin: 159 [160, 171)
"""

RAW_CLASSES = {"Wild type": 0,
               "GABAA pore blocker": 1,
               "vesicular ACh transport antagonist": 2,
               "nAChR orthosteric agonist": 3,
               "nAChR orthosteric antagonist": 4,
               "TRPV agonist": 5,
               "GABAA allosteric antagonist": 6,
               "RyR agonist": 7,
               "Na channel": 8,
               "complex II inhibitor": 9,
               "nAChR allosteric agonist": 10,
               "unknown-likely neurotoxin": 11
               }

control_name= "all" # ""Control_12

def get_key_dict_list(dict_list, value):
    for k, v_list in dict_list.items():
        for v in v_list:
            if v == value:
                return k
    return "None"

def manova(data={}):
    from statsmodels.multivariate.manova import MANOVA
    data = pd.DataFrame.from_dict(data)
    print(data)

    fit = MANOVA.from_formula('height + canopy_vol ~ plant_var', data=data)
    print(fit.mv_test())

def anova(data={}) -> np.array:
    keys = data.keys()
    num_keys = len(keys)
    anova_metrix = np.zeros((num_keys+1,num_keys+1))
    anova_metrix[1:, 0] = range(1, num_keys+1)
    anova_metrix[0, 1:] = range(1, num_keys+1)
    anova_metrix[0, 0] = None

    for i in list(anova_metrix[1:, 0]):
        control_a = data["Control_"+str(int(i))]
        for j in list(anova_metrix[0, 1:]):
            control_b = data["Control_" + str(int(j))]
            print(i, j)
            res = pg.multivariate_ttest(control_a, Y=control_a)
            print(res)
            break
        break

def compare_controls_to(action_path, data_path, save_path):
    # read action names
    all_control_data = {}
    # read controls
    control_path = data_path / "Controls/"
    control_folders = os.listdir(control_path)
    num_controls = len(control_folders)

    invalid = 0
    for c_i, c_folder in enumerate(control_folders):
        if c_folder[0] == ".":
            num_controls = num_controls-1
            invalid+=1
            continue
        all_control_data[c_folder] = []
        c_folder_path = control_path / c_folder
        control_files = os.listdir(c_folder_path)
        for c_f in control_files:
            print(c_folder_path / c_f)
            with open(c_folder_path / c_f, "r") as control_d_f:
                reader_to_lines = []
                reader = csv.reader(control_d_f, delimiter=",")
                for j, l in enumerate(reader):
                    reader_to_lines.append([float(i) for i in l])
                reader_to_lines = np.array(reader_to_lines)
                for i in range(reader_to_lines.shape[1]):
                    all_control_data[c_folder].append(reader_to_lines[:, i].tolist())
    print("number of controls", num_controls)
    anova(all_control_data)
    """
    with open(save_path / ("controls_"+control_name+".csv"), "w") as save_csv:
        writer = csv.writer(save_csv)
        for key, value in all_control_data.items():
            for v in value:
                writer.writerow([key] + v)
    """


if __name__ == "__main__":
    action_path = "./data/OldCompoundsMoA.csv"
    data_path = TRAIN_SAVE_CLEAN_PATH / "all_data/"
    save_path = TRAIN_SAVE_FINAL_RESULT_PATH
    compare_controls_to(action_path, data_path, save_path)