import matplotlib.pyplot as plt
import csv
import numpy as np
from utils import RAW_CLASSES, load_feature_data_all_by_compound
from scipy.stats import ttest_ind

def t_test_visualize_save(data={}, action={}, p_thre=5, save_path=""):
    compound_names = data.keys()
    selected_compounds = []
    selected_data = []
    wt_data = np.array(data["C0"])
    for w_d in wt_data:
        #print(w_d)
        selected_data.append(["C0"] + w_d.tolist() + ["Wild type"] + [0])
    for comp_name in compound_names:
        if comp_name != "C0":
            comp_data = data[comp_name]
            comp_data = np.array(comp_data)
            F, p = ttest_ind(wt_data, comp_data, equal_var=False)
            p_greater = np.sum(p<0.05)
            if p_greater > p_thre:
                selected_compounds.append(comp_name)
                for c_d in comp_data:
                    selected_data.append([comp_name] + c_d.tolist() + action[comp_name])

            fig, axs = plt.subplots(2, figsize=(30, 10))

            # axs[a].set_size_inches(30, 10)
            axs[0].boxplot(wt_data, labels=range(22), widths=0.5, positions=range(wt_data.shape[1]),
                           patch_artist=True,
                           showfliers=True, showmeans=True)
            axs[0].set_ylabel("Feature")
            # axs[0].set_xlabel("Feature index")
            axs[0].set_title("Wild type")
            axs[0].set_ylim(0, 1)
            axs[0].margins(x=0)

            axs[1].boxplot(comp_data, labels=range(22), widths=0.5, positions=range(comp_data.shape[1]),
                           patch_artist=True,
                           showfliers=True, showmeans=True)
            axs[1].set_ylabel("Feature")
            # axs[1].set_xlabel("Feature index")
            axs[1].set_title(comp_name)
            axs[1].set_ylim(0, 1)
            axs[1].margins(x=0)
            # plt.grid(b=True, which="both", axis="both")
            # plt.ylabel(ylabels[2], fontsize=8)
            # plt.title(titles[2])
            # plt.xticks(fontsize=14, fontname="Times New Roman")
            # plt.yticks(fontsize=14, fontname="Times New Roman")
            # plt.title("Wild type VS " + comp_name, fontname="Times New Roman", fontsize=14)
            for i, (f_v, p_v) in enumerate(zip(F, p)):
                #axs[0].text(i, 0.108, s="F:" + str(round(f_v, 4)), color="tab:red")
                axs[0].text(i, 1.01, s="p:" + str(round(p_v, 4)), color="tab:red")
            plt.tight_layout()
            # plt.show()
            fig.savefig(save_path + "t_test_wildtype_vs_" + comp_name + ".png")

    with open("/srv/yanke/PycharmProjects/HTScreening/data/effected/effected_compounds_feature_max_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(selected_data)
    print("effected compounds are: ", selected_compounds)



if __name__ == "__main__":
    feature_data_by_compound, action_dict_by_compound = load_feature_data_all_by_compound(path="/srv/yanke/PycharmProjects/HTScreening/data/featured/all_compounds_feature_max_fish_with_action.csv")
    t_test_visualize_save(feature_data_by_compound, action_dict_by_compound, 20, save_path="./t_test_all_feature_max/")
