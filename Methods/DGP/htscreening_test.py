import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import time
from gpflow.kernels import RBF
from gpflow.likelihoods import MultiClass, Bernoulli
from scipy.cluster.vq import kmeans2
from scipy.stats import mode
from data_loader import load_cleaned_data, RAW_CLASSES, get_key
from deep_gp import DeepGP
from htscreening import make_dgp
from utils import plot_tsne, plot_latent_heatmap_by_compound, plot_tsne_no_class, \
    plot_tsne_by_compound, plot_tsne_train_eval_by_compound, \
    plot_2Ddistribution_train_eval_by_compound, plot_2Ddistribution_train_eval_by_action_mode
from feature_selection import median

LATENT_DIMENSION = 2
NUMBER_LAYERS = 4
MATH_PATH = "./results/dgp_vae/22-dimension/"


def test_on_train_eval(x_train, x_eval, y_train, train_labels, eval_labels, num_inducing=130, batch_size=100, num_samples = 1):
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    #num_samples = 1  # guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(NUMBER_LAYERS, x_train, y_train, Z)
    checkpoint = tf.train.Checkpoint(model=dgp)
    # save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, MATH_PATH + "models/", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    # run the network
    labels = {}
    labels["train"] = train_labels
    labels["eval"] = eval_labels
    latent_train_eval = {}
    for X, k in zip([x_train, x_eval], ["train", "eval"]):
        n_batches = max(int(len(X) / batch_size), 1)
        print(n_batches)
        latent_ms = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
        for n_b in range(n_batches + 1):
            x_batch = X[(n_b * batch_size):((n_b + 1) * batch_size), :]
            m_each_layer, v_each_layer = dgp.predict_each_layer(x_batch, num_samples)
            # print(m_each_layer[-2].shape)
            latent_ms[(n_b * batch_size):((n_b + 1) * batch_size), :] = m_each_layer[-1*NUMBER_LAYERS//2-1][0, :, :]
        latent_train_eval[k] = latent_ms

    return latent_train_eval, labels

def save_latent_feature(x_train, train_comps, train_labels, num_inducing=130, num_samples = 1, save_path=""):
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    # num_samples = 1  # guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(NUMBER_LAYERS, x_train, y_train, Z)
    checkpoint = tf.train.Checkpoint(model=dgp)
    # save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, MATH_PATH + "models/", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    # run the network
    all_data_feature = []
    for x, c, l in zip(x_train, train_comps, train_labels):
        m_each_layer, v_each_layer = dgp.predict_each_layer(x, num_samples)
        latent_ms = m_each_layer[-1 * NUMBER_LAYERS // 2 - 1][0, :,:]
        feature_data = [c] + latent_ms + [get_key(RAW_CLASSES, l), l]
        all_data_feature.append(feature_data)
    with open(save_path + "all_compounds_DGP_VAE_feature_fish_with_action.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(all_data_feature)

def draw_tsne(model, X, labels, batch_size=1000, num_samples=100):
    #num_layers = len(model.layers)
    #print(num_layers, model.num_samples)
    n_batches = max(int(len(X) / batch_size), 1)
    print(n_batches)
    latent_ms = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
    for n_b in range(n_batches+1):
        x_batch = X[(n_b*batch_size):((n_b+1)*batch_size), :]
        m_each_layer, v_each_layer = model.predict_each_layer(x_batch, num_samples)
        #print(m_each_layer[-2].shape)
        latent_ms[(n_b*batch_size):((n_b+1)*batch_size), :] = m_each_layer[-1*NUMBER_LAYERS//2-1][0, :, :]
    #labels = np.squeeze(labels, axis=1)
    #print(labels.shape)
    #plot_latent_heatmap_by_compound(latent_ms, labels, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    #plot_tsne_no_class(latent_ms, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    print(labels)
    plot_tsne_by_compound(latent_ms, labels, name="dgp_htscreening", save_path="./results/dgp_vae/train/compounds/")

if __name__ == '__main__':
    x_train, train_comps, train_labels = load_cleaned_data(
        path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv")
    save_latent_feature(x_train, train_comps, train_labels, num_inducing=130, num_samples=1,
                        save_path="/srv/yanke/PycharmProjects/HTScreening/data/median_clustering/")
    """
    action_dict = load_action_mode(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    print(action_dict)
    y_train = x_train
    y_eval = x_eval
    latent_train_eval, labels = test_on_train_eval(x_train, x_eval, y_train, train_label, eval_label, num_inducing=130, batch_size=100, num_samples = 1)
    plot_2Ddistribution_train_eval_by_compound(latent_train_eval, labels, name="dgp_htscreening",
                                               save_path=MATH_PATH+"all/compounds/")
    feature_new, labels_new = median(latent_train_eval, labels, num_class=130)
    plot_2Ddistribution_train_eval_by_compound(feature_new, labels_new, name="dgp_htscreening",
                                               save_path=MATH_PATH+"all/compounds_feature_select/")
    plot_2Ddistribution_train_eval_by_action_mode(feature_new, labels_new, action_dict, name="pca_htscreening",
                                                  save_path=MATH_PATH+"all/actions/")
    """