import os
import shutil

def rename_file_to(ori_path, save_path):
    file_names = os.listdir(path=ori_path)
    for f_n in file_names:
        f_n_split = f_n[:-4].split("_")
        if f_n_split[2][-1] == "M":
            if f_n_split[1][0] == "C":
                save_name = f_n_split[1] + "_50um_" + f_n_split[0] + ".csv"
            else:
                save_name = "C" + f_n_split[1] + "_50um_" + f_n_split[0] + ".csv"
        else:
            if f_n_split[2][0] == "C":
                save_name = f_n_split[2] + "_50um_" + f_n_split[0] + ".csv"
            else:
                save_name = "C" + f_n_split[2] + "_50um_" + f_n_split[0] + ".csv"

        shutil.copy(ori_path+f_n, save_path+save_name)


if __name__ == "__main__":
    file_path = "./data/ori/all_data/Compounds/"
    save_path = "./data/ori/all_data/Compounds_renamed/"
    rename_file_to(file_path, save_path)