import numpy as np
from Methods.DGP.data_loader import load_cleaned_data
from Methods.DGP.htscreening_test import test_on_train_eval
from Methods.PCA.pca_dim_reduce import try_PCA_with_test


def Covariance(feature):

    return np.std(feature)

def cov_estimate(feature={}, labels={}, num_class = 130):
    ave_cov_train = 0
    ave_cov_eval = 0
    i = 0
    for c in range(num_class):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = feature["train"][train_inds]
        eval_comp_data = feature["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue
        i += 1

        ave_cov_train += Covariance(train_comp_data)
        ave_cov_eval += Covariance(eval_comp_data)

    return ave_cov_train, ave_cov_eval




def Evaluation(method=["PCA", "VAE", "DGP"]):
    x_train, train_label = load_cleaned_data(
        path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv",
        label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_label.csv",
        normalize=False)

    x_eval, eval_label = load_cleaned_data(
        path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_set.csv",
        label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_label.csv",
        normalize=False)
    y_train = x_train
    y_eval = x_eval

    for m in method:
        if m == "PCA":
            labels = {}
            labels["train"] = train_label
            labels["eval"] = eval_label
            latent_train_eval = {}
            new_train, new_eval = try_PCA_with_test(x_train, x_eval)
            latent_train_eval["train"] = new_train
            latent_train_eval["eval"] = new_eval
            train_cov, test_cov = cov_estimate(latent_train_eval, labels)
            print(train_cov, test_cov)
        elif m == "VAE":
            pass
        elif m == "DGP":
            latent_train_eval, labels = test_on_train_eval(x_train, x_eval, y_train, train_label, eval_label,
                                                           num_inducing=130, batch_size=100, num_samples=1)
            train_cov, test_cov = cov_estimate(latent_train_eval, labels)
            print(train_cov, test_cov)

if __name__ == "__main__":
    Evaluation(method=["DGP"])