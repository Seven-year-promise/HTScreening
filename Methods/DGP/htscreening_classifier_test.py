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
from data_loader import load_cleaned_data, load_action_mode, load_effected_data
from deep_gp import DeepGP
from htscreening_classifier import make_dgp
from utils import plot_tsne, plot_latent_heatmap_by_compound, plot_tsne_no_class, \
    plot_tsne_by_compound, plot_tsne_train_eval_by_compound, \
    plot_2Ddistribution_train_eval_by_compound, plot_2Ddistribution_train_eval_by_action_mode, plot_tsne_by_action
from feature_selection import median

LATENT_DIMENSION = 30
NUMBER_LAYERS = 4
MATH_PATH = "./results/dgp_vae/class/split/30-dimension/"


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

def test_on_train(model, X, labels, batch_size=100, num_samples = 1):
    n_batches = max(int(len(X) / batch_size), 1)
    print(n_batches)
    latent_ms = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
    likelihood = np.zeros((len(X)), dtype=np.float)
    acc = np.zeros((len(X), 1), dtype=np.float)
    for n_b in range(n_batches + 1):
        if n_b == n_batches:
            x_batch = X[(n_b * batch_size):, :]
            y_batch = labels[(n_b * batch_size):, :]
        else:
            x_batch = X[(n_b*batch_size):((n_b+1)*batch_size), :]
            y_batch = labels[(n_b*batch_size):((n_b+1)*batch_size), :]
        # get the latent space
        m_each_layer, v_each_layer = model.predict_each_layer(x_batch, num_samples)
        # print(m_each_layer[-2].shape)
        latent_ms[(n_b * batch_size):((n_b + 1) * batch_size), :] = m_each_layer[-2][0, :, :]

        # get the accuracy
        m, v = model.predict_y(x_batch, num_samples)
        likelihood[(n_b * batch_size):((n_b + 1) * batch_size)] = model.predict_log_density((x_batch, y_batch),
                                                                                            num_samples)
        acc[(n_b * batch_size):((n_b + 1) * batch_size), :] = (
                mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))

    return latent_ms, np.mean(likelihood), np.mean(acc)

def evaluation_step(model, X, Y, batch_size=1000, num_samples=100):
    n_batches = max(int(len(X) / batch_size), 1)
    likelihoods, accs = [], []
    for x_batch, y_batch in zip(np.split(X, n_batches),
                                np.split(Y, n_batches)):
        m, v = model.predict_y(x_batch, num_samples)
        likelihood = model.predict_log_density((x_batch, y_batch), num_samples)
        acc = (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
        likelihoods.append(likelihood)
        accs.append(acc)
    return np.mean(likelihoods), np.mean(accs)

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
    x_train, y_train = load_effected_data(
        path="/srv/yanke/PycharmProjects/HTScreening/data/effected_dataset/train_set.csv",
        normalize=False)
    batch_size = 100
    num_inducing = 200
    num_samples = 1
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    # num_samples = 1  # guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(NUMBER_LAYERS, x_train, y_train, Z)
    checkpoint = tf.train.Checkpoint(model=dgp)
    # save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, MATH_PATH + "models/", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)


    latent_train, likelihood, acc = test_on_train(dgp, x_train, labels=y_train, batch_size=100, num_samples = 1)
    print(f"Likelihood: {likelihood}, Acc: {acc} ")
    plot_tsne_by_action(latent_train, y_train, name="dgp_htscreening_class_train", save_path=MATH_PATH+"all/")