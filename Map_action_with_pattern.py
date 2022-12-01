import numpy as np
import nthresh
import csv
import sys
sys.path.append("../")
from utils import RAW_CLASSES, load_feature_data_all_by_compound
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from utils import read_mode_action
from config import *

"""
visualize the intergrated feature
"""
p_thre=0.05
feature_num = 3
type = "integration"


def map_and_save(path, actions, save_path): # actions format -> dict -> action name: [compound1, compound2] without 'C'
    action_effect_codes = [] # # actions format -> list -> action name: [[compound1, effect code1, effect code2], [compound2, effect code1, effect code2],,,,]
    effect_codes = {} # format -> dict -> compound name (with 'C'): effect_code
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            if j < 1:
                continue
            compound_name = l[0]
            effect_codes[compound_name] = int(l[-1])

    for a_k in actions.keys():
        list_compounds = actions[a_k]
        if len(list_compounds)>0:
            list_effect_codes = []
            for c_n in list_compounds:

                c_k = "C" + str(c_n)
                if c_k in effect_codes.keys():
                    list_effect_codes.append(effect_codes[c_k])
            list_effect_codes = list(set(list_effect_codes)) # delete the duplicate
            action_effect_codes.append([a_k]+list_effect_codes)



    print(action_effect_codes)

    with open(save_path / ("action_with_effect_codes_"+str(feature_num)+type+str(p_thre)+".csv"), "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(action_effect_codes)



if __name__ == "__main__":
    #save_binary_code_mapping_motion(
    #    binary_path="/Users/yankeewann/Desktop/HTScreening/data/featured/effects_binary_codes_with_integration.csv",
    #    save_path="/Users/yankeewann/Desktop/HTScreening/data/")

    actins = read_mode_action()
    map_and_save(SAVE_FEATURE_PATH / ("effects_binary_codes_with_"+str(feature_num)+type + str(p_thre)+".csv"), actins, SAVE_FINAL_RESULT_PATH)

