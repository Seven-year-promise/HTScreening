import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, "/srv/yanke/PycharmProjects/HTScreening")
import numpy as np
import tensorflow as tf

import time

from gpflow.kernels import RBF
from gpflow.likelihoods import MultiClass
from scipy.cluster.vq import kmeans2
from scipy.stats import mode

from deep_gp import DeepGP
import csv


def load_data(data_path, label_path):

    data_list = []
    data_labels = []
    with open(data_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_line = [float(i) for i in l]
            data_list.append(data_line)

    with open(label_path, newline='') as csv_f:
        read_lines = csv.reader(csv_f, delimiter=",")
        for j, l in enumerate(read_lines):
            data_labels = [int(i) for i in l]

    x = np.array(data_list)
    y = np.array(data_labels).reshape(-1, 1)
    return x, y

def make_dgp(num_layers, X, Y, Z):
    kernels = [RBF(variance=2.0, lengthscales=2.0)]
    layer_sizes = [568]
    for l in range(num_layers-1):
        kernels.append(RBF(variance=2.0, lengthscales=2.0))
        layer_sizes.append(30)
    model = DeepGP(X, Y, Z, kernels, layer_sizes, MultiClass(4),
                   num_outputs=4)

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
        acc = (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
        likelihoods.append(likelihood)
        accs.append(acc)
    return np.mean(likelihoods), np.mean(accs)


if __name__ == '__main__':
    print(sys.path)
    x_train, y_train = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_set.csv",
                                 label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/train_label.csv")
    x_test, y_test = load_data(data_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/test_set.csv",
                               label_path="/srv/yanke/PycharmProjects/HTScreening/data/dataset/test_label.csv")

    x_train, y_train = x_train[:3000], y_train[:3000]
    x_test, y_test = x_test[:800], y_test[:800]

    print(x_train.shape, y_train.shape, x_test.shape)
    num_inducing = 100
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    batch_size = 200
    num_samples = 100

    dgp = make_dgp(2, x_train, y_train, Z)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    for _ in range(1500):
        start_time = time.time()
        elbo = training_step(dgp, x_train, y_train, batch_size)
        likelihood, acc = evaluation_step(dgp, x_train, y_train, batch_size,
                                          num_samples)
        duration = time.time() - start_time
        print(f"ELBO: {elbo}, Likelihood: {likelihood}, Acc: {acc} [{duration}]")
