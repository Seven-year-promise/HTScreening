from data_loader import load_cleaned_data
from utils import plot_tsne_train_eval_by_compound, plot_2Ddistribution_train_eval_by_compound, plot_2Ddistribution_train_eval_by_action_mode
from pca_dim_reduce import try_PCA_with_test
from feature_selection import median
from data_loader import load_action_mode

LATENT_DIMENSION = 2

MATH_PATH = "./results/cleaned_data/all-2D/"


if __name__ == '__main__':
    x_train, train_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv",
                          label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_label.csv",
                          normalize=False)

    x_eval, eval_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_set.csv",
                          label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_label.csv",
                          normalize=False)
    y_train = x_train
    y_eval = x_eval

    action_dict = load_action_mode(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    print(action_dict)

    labels = {}
    labels["train"] = train_label
    labels["eval"] = eval_label
    latent_train_eval = {}
    new_train, new_eval = try_PCA_with_test(x_train, x_eval)
    latent_train_eval["train"] = new_train
    latent_train_eval["eval"] = new_eval

    plot_2Ddistribution_train_eval_by_compound(latent_train_eval, labels, name="pca_htscreening",
                                               save_path=MATH_PATH + "pca/compounds/")

    feature_new, labels_new = median(latent_train_eval, labels, num_class=130)

    plot_2Ddistribution_train_eval_by_compound(feature_new, labels_new, name="pca_htscreening",
                                               save_path=MATH_PATH + "pca_feature_select/compounds/")
    plot_2Ddistribution_train_eval_by_action_mode(feature_new, labels_new, action_dict, name="pca_htscreening",
                                               save_path=MATH_PATH + "pca_feature_select/actions/")