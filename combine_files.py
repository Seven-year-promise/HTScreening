import os
import shutil
import csv
import numpy as np
from config import *

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

def combine_file_to(action_path, data_path, save_path):
    # read action names
    action_with_compounds = {}
    for a in RAW_CLASSES.keys():
        action_with_compounds[a] = []
    with open(action_path, "r") as a_f:
        reader_to_lines = []
        reader = csv.reader(a_f, delimiter=",")
        for j, l in enumerate(reader):
            #print(l[0])
            reader_to_lines.append(l[0])
        action_with_compounds[reader_to_lines[1]] = [int(i) for i in reader_to_lines[2:30]]
        action_with_compounds[reader_to_lines[30]] = [int(i) for i in reader_to_lines[31:43]]
        action_with_compounds[reader_to_lines[43]] = [int(i) for i in reader_to_lines[44:52]]
        action_with_compounds[reader_to_lines[52]] = [int(i) for i in reader_to_lines[53:63]]
        action_with_compounds[reader_to_lines[63]] = [int(i) for i in reader_to_lines[64:122]]
        action_with_compounds[reader_to_lines[122]] = [int(i) for i in reader_to_lines[123:144]]
        action_with_compounds[reader_to_lines[144]] = [int(i) for i in reader_to_lines[145:147]]
        action_with_compounds[reader_to_lines[147]] = [int(i) for i in reader_to_lines[148:150]]
        action_with_compounds[reader_to_lines[150]] = [int(i) for i in reader_to_lines[151:156]]
        action_with_compounds[reader_to_lines[156]] = [int(i) for i in reader_to_lines[157:159]]
        action_with_compounds[reader_to_lines[159]] = [int(i) for i in reader_to_lines[160:171]]
    print("compound action modes:", action_with_compounds)
    all_control_comp_data = []
    # read controls
    control_path = data_path / "Controls/"
    control_folders = os.listdir(control_path)
    num_controls = len(control_folders)
    print("number of controls", num_controls)
    invalid = 0
    for c_i, c_folder in enumerate(control_folders):
        if c_folder[0] == ".":
            num_controls = num_controls-1
            invalid+=1
            continue
        print(c_i)
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
                    one_control_data = ["C0"] + reader_to_lines[:, i].tolist() + ["Wild type"] + [0]#["C"+str(c_i-invalid)] + reader_to_lines[:, i].tolist() + ["Wild type"] + [0]
                    all_control_comp_data.append(one_control_data)

    # read compounds
    comp_path = data_path / "Compounds/"
    comp_files = os.listdir(comp_path)
    for comp_f in comp_files:
        print(comp_path / comp_f)
        with open(comp_path / comp_f, "r") as comp_d_f:
            comp_name = comp_f[:-4].split("_")[0]
            comp_id = int(comp_name[1:])
            comp_name = "C"+str(comp_id)#str(num_controls-1+comp_id)
            reader_to_lines = []
            reader = csv.reader(comp_d_f, delimiter=",")
            for j, l in enumerate(reader):
                reader_to_lines.append([float(i) for i in l])
            reader_to_lines = np.array(reader_to_lines)
            for i in range(reader_to_lines.shape[1]):
                action_name = get_key_dict_list(action_with_compounds, comp_id)
                action_id = RAW_CLASSES[action_name]
                one_comp_data = [comp_name] + reader_to_lines[:, i].tolist() + [action_name] + [action_id]
                all_control_comp_data.append(one_comp_data)

    with open(save_path / ("all_compounds_ori_fish_with_action_wt_"+control_name+".csv"), "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_control_comp_data)


if __name__ == "__main__":
    action_path = "./data/OldCompoundsMoA.csv"
    data_path = SAVE_CLEAN_PATH / "all_data/"
    save_path = SAVE_CLEAN_PATH
    combine_file_to(action_path, data_path, save_path)