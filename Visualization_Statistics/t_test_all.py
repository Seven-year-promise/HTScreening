import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from Analysis import f_one_way_test
from utils import RAW_CLASSES, load_feature_data_all_by_compound
from scipy.stats import ttest_ind

def t_test_visualize_save(data={}, p_thre=5, save_path=""):
    compound_names = data.keys()
    selected_compounds = []
    wt_data = np.array(data["C0"])
    for comp_name in compound_names:
        if comp_name != "C0":
            comp_data = np.array(data[comp_name])
            f, p = ttest_ind(wt_data, comp_data, equal_var=False)
            p_greater = np.sum(p<0.05)
            if p_greater > p_thre:
                selected_compounds.append(comp_name)

            fig, axs = plt.subplots(2, figsize=(30, 10))

            # axs[a].set_size_inches(30, 10)
            axs[0].boxplot(wt_data, labels=np.range(22), widths=0.5, positions=range(action_data.shape[1]),
                           patch_artist=True,
                           showfliers=True, showmeans=True)
            axs[0].set_ylabel("Wild type")
            axs[0].set_xlabel("Feature index")
            axs[0].set_title(get_key(CLASSES, a))
            axs[0].set_ylim(0, 0.1)
            axs[0].margins(x=0)

            axs[0].boxplot(comp_data, labels=np.range(22), widths=0.5, positions=range(action_data.shape[1]),
                           patch_artist=True,
                           showfliers=True, showmeans=True)
            axs[0].set_ylabel(comp_name)
            axs[0].set_xlabel("Feature index")
            axs[0].set_title(get_key(CLASSES, a))
            axs[0].set_ylim(0, 0.1)
            axs[0].margins(x=0)
            # plt.grid(b=True, which="both", axis="both")
            # plt.ylabel(ylabels[2], fontsize=8)
            # plt.title(titles[2])
            # plt.xticks(fontsize=14, fontname="Times New Roman")
            # plt.yticks(fontsize=14, fontname="Times New Roman")
            plt.title("Wild type VS " + comp_name, fontname="Times New Roman", fontsize=14)
            for i, (f_v, p_v) in enumerate(zip(F, p)):
                #axs[0].text(i, 0.108, s="F:" + str(round(f_v, 4)), color="tab:red")
                axs[0].text(i, 0.101, s="p:" + str(round(p_v, 4)), color="tab:red")
            plt.tight_layout()
            # plt.show()
            fig.savefig(save_path + "t_test_wildtype_vs_" + comp_name + ".png")

    print("effected compounds are: ", selected_compounds)



if __name__ == "__main__":
    feature_data_by_compound = load_feature_data_all_by_compound(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    visualize_action_box_plot(data, actions, 4, save_path="./feature_median_14/")
