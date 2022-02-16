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

from deep_gp import DeepGP
from utils import plot_tsne, plot_latent_heatmap_by_compound, plot_tsne_no_class, plot_tsne_by_compound, plot_tsne_train_eval_by_compound

def load_data():
    (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train = (x_train.astype(np.float64) / 255.0) - 0.5
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_eval = (x_eval.astype(np.float64) / 255.0) - 0.5
    x_eval = x_eval.reshape(x_eval.shape[0], -1)
    y_train = y_train.reshape(-1, 1)
    y_eval = y_eval.reshape(-1, 1)
    return x_train, y_train, x_eval, y_eval

def load_cleaned_data(path, label_path, normalize):
    data_list = []
    data_labels = []
    with open(path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)

    with open(label_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_labels = [int(i) for i in l]

    data_list = np.array(data_list)
    if normalize:
        transformer = Normalizer().fit(np.array(data_list))
        data_list = transformer.transform(data_list)

    return data_list, np.array(data_labels)

def make_dgp(num_layers, X, Y, Z):
    kernels = [RBF(variance=2.0, lengthscales=2.0)]
    layer_sizes = [540]
    for l in range(num_layers-1):
        kernels.append(RBF(variance=2.0, lengthscales=2.0))
        layer_sizes.append(30)
    model = DeepGP(X, Y, Z, kernels, layer_sizes, Bernoulli(),
                   num_outputs=540)

    # init hidden layers to be near deterministic
    for layer in model.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    return model


def training_step(model, X, Y, batch_size=1000):
    n_batches = max(int(len(x_train) / batch_size), 1)
    elbos = []
    for x_batch, y_batch in zip(np.split(X, n_batches),
                                np.split(Y, n_batches)):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(dgp.trainable_variables)
            objective = -model.elbo((x_batch, y_batch))
            gradients = tape.gradient(objective, dgp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dgp.trainable_variables))
        elbos.append(-objective.numpy())
    return np.mean(elbos)


def evaluation_step(model, X, Y, batch_size=1000, num_samples=100):
    n_batches = max(int(len(X) / batch_size), 1)
    likelihoods, accs = [], []
    for x_batch, y_batch in zip(np.split(X, n_batches),
                                np.split(Y, n_batches)):
        m, v = model.predict_y(x_batch, num_samples)
        likelihood = model.predict_log_density((x_batch, y_batch), num_samples)
        acc = 0 # (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
        likelihoods.append(likelihood)
        accs.append(acc)
    return np.mean(likelihoods), np.mean(accs)

def draw_tsne(model, X, labels, batch_size=1000, num_samples=100):
    #num_layers = len(model.layers)
    #print(num_layers, model.num_samples)
    n_batches = max(int(len(X) / batch_size), 1)
    print(n_batches)
    latent_ms = np.zeros((len(X), 30), dtype=np.float)
    for n_b in range(n_batches+1):
        x_batch = X[(n_b*batch_size):((n_b+1)*batch_size), :]
        m_each_layer, v_each_layer = model.predict_each_layer(x_batch, num_samples)
        #print(m_each_layer[-2].shape)
        latent_ms[(n_b*batch_size):((n_b+1)*batch_size), :] = m_each_layer[-2][0, :, :]
    #labels = np.squeeze(labels, axis=1)
    #print(labels.shape)
    #plot_latent_heatmap_by_compound(latent_ms, labels, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    #plot_tsne_no_class(latent_ms, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    print(labels)
    plot_tsne_by_compound(latent_ms, labels, name="dgp_htscreening", save_path="./results/dgp_vae/train/compounds/")

def draw_train_eval_tsne(model, X_train, X_eval, train_labels, eval_labels, batch_size=1000, num_samples=100):
    #num_layers = len(model.layers)
    #print(num_layers, model.num_samples)
    labels = {}
    labels["train"] = train_labels
    labels["eval"] = eval_labels
    latent_train_eval = {}
    for X, k in zip([X_train, X_eval], ["train", "eval"]):
        n_batches = max(int(len(X) / batch_size), 1)
        print(n_batches)
        latent_ms = np.zeros((len(X), 30), dtype=np.float)
        for n_b in range(n_batches+1):
            x_batch = X[(n_b*batch_size):((n_b+1)*batch_size), :]
            m_each_layer, v_each_layer = model.predict_each_layer(x_batch, num_samples)
            #print(m_each_layer[-2].shape)
            latent_ms[(n_b*batch_size):((n_b+1)*batch_size), :] = m_each_layer[-2][0, :, :]
        latent_train_eval[k] = latent_ms
    #labels = np.squeeze(labels, axis=1)
    #print(labels.shape)
    #plot_latent_heatmap_by_compound(latent_ms, labels, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    #plot_tsne_no_class(latent_ms, name="dgp_htscreening", save_path="./results/dgp_vae/eval/")
    #print(labels)
    plot_tsne_train_eval_by_compound(latent_train_eval, labels, name="dgp_htscreening", save_path="./results/dgp_vae/all/compounds/")

if __name__ == '__main__':
    x_train, train_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv",
                          label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_label.csv",
                          normalize=False)

    x_eval, eval_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_set.csv",
                          label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_label.csv",
                          normalize=False)
    y_train = x_train
    y_eval = x_eval
    #y_train, x_train, y_eval, x_eval, eval_label = y_train[:1000, :], x_train[:1000, :], y_eval[:1000, :], x_eval[:1000, :], eval_label[:1000, :]
    #for i in range(1000):
    #    plt.plot(x_train[i, :])
    #plt.show()
    num_inducing = 130
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    batch_size = 100
    num_samples = 1 #guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(2, x_train, y_train, Z)
    checkpoint = tf.train.Checkpoint(model=dgp)
    # save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, "./results/dgp_vae/models/", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    # tf.saved_model.save(dgp, "./results/dgp_vae/dgp_vae/")
    print(x_eval.shape, eval_label.shape)
    #draw_tsne(dgp, x_train, train_label, batch_size, num_samples)
    draw_train_eval_tsne(dgp, x_train, x_eval, train_label, eval_label, batch_size, num_samples)