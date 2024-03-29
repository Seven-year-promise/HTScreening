import os
import shutil
import csv
from config import *

"""
clean rule

[3:190) before the first light
[204:380) after the first light
[395:) after the second light
"""

def clean_file_to(ori_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_names = ori_path.rglob("*csv")
    for f_n in file_names:
        with open(f_n, "r") as l_f:
            reader_to_lines = []
            reader = csv.reader(l_f, delimiter=",")
            for j, l in enumerate(reader):
                reader_to_lines.append([float(i) for i in l])
            data_to_save = reader_to_lines[3:190] + reader_to_lines[204:380] + reader_to_lines[395:]
        with open(save_path / f_n.name, "w") as save_csv:
            csv_writer = csv.writer(save_csv)
            csv_writer.writerows(data_to_save)


if __name__ == "__main__":
    file_path = TEST_ORI_PATH / "all_data/Compounds_renamed/"  #/Compounds_renamed/"# /Controls/Control 1, 2, 3, 4_candidates....
    save_path = TEST_SAVE_CLEAN_PATH / "all_data/Compounds/" #/Compounds/" # /Controls/Control_1, 2, 3, ....
    clean_file_to(file_path, save_path)