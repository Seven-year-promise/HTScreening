import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
import os
import time
from data_loader import load_cleaned_data, load_effected_data, load_filtered_effected_data, load_effected_action_data, load_effected_action_data_dimension
from gpflow.kernels import RBF
from gpflow.likelihoods import MultiClass, Bernoulli
from scipy.cluster.vq import kmeans2
from scipy.stats import mode

from deep_gp import DeepGP
from utils import plot_tsne, plot_tsne_no_class

LATENT_DIMENSION = 30
MATH_PATH = "./results/dgp_vae/effected/class/ori/binary_all/"
NUMBER_LAYERS = 4

def make_dgp(num_layers, X, Y, Z):
    kernels = [RBF(variance=2.0, lengthscales=2.0)]
    layer_sizes = [541, 256, 128, LATENT_DIMENSION]
    for l in range(num_layers-1):
        kernels.append(RBF(variance=2.0, lengthscales=2.0))
    model = DeepGP(X, Y, Z, kernels, layer_sizes, MultiClass(2),
                   num_outputs=2)

    # init hidden layers to be near deterministic
    for layer in model.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    return model


def training_step(model, X, Y, batch_size=1000):
    n_batches = max(int(len(X) / batch_size), 1)
    elbos = []
    for n_b in range(n_batches+1):
        patch_choice = np.random.choice(len(X), batch_size)
        """
        if n_b == n_batches:
            x_batch = X[(n_b * batch_size):, :]
            y_batch = Y[(n_b * batch_size):, :]
        else:
            x_batch = X[(n_b*batch_size):((n_b+1)*batch_size), :]
            y_batch = Y[(n_b*batch_size):((n_b+1)*batch_size), :]
        """
        x_batch = X[patch_choice, :]
        y_batch = Y[patch_choice, :]
        #print(x_batch.shape, y_batch.shape)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(dgp.trainable_variables)
            objective = -model.elbo((x_batch, y_batch))
            gradients = tape.gradient(objective, dgp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dgp.trainable_variables))
        elbos.append(-objective.numpy())
    return np.mean(elbos)


def evaluation_step(model, X, Y, batch_size=1000, num_samples=100):
    n_batches = max(int(len(X) / batch_size), 1)
    likelihood = np.zeros((len(X)), dtype=np.float)
    acc = np.zeros((len(X), 1), dtype=np.float)
    for n_b in range(n_batches + 1):
        if n_b == n_batches:
            x_batch = X[(n_b * batch_size):, :]
            y_batch = Y[(n_b * batch_size):, :]
            m, v = model.predict_y(x_batch, num_samples)
            likelihood[(n_b * batch_size):] = model.predict_log_density((x_batch, y_batch), num_samples)
            acc[(n_b * batch_size):, :] = (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
        else:
            x_batch = X[(n_b * batch_size):((n_b + 1) * batch_size), :]
            y_batch = Y[(n_b * batch_size):((n_b + 1) * batch_size), :]
            m, v = model.predict_y(x_batch, num_samples)
            likelihood[(n_b * batch_size):((n_b + 1) * batch_size)] = model.predict_log_density((x_batch, y_batch), num_samples)
            acc[(n_b * batch_size):((n_b + 1) * batch_size), :] = (
                        mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))

    return np.mean(likelihood), np.mean(acc)

def draw_tsne(model, X, labels, batch_size=1000, num_samples=100):
    #num_layers = len(model.layers)
    #print(num_layers, model.num_samples)
    n_batches = max(int(len(X) / batch_size), 1)
    likelihoods, accs = [], []
    latent_ms = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
    for n_b, (x_batch, y_batch) in enumerate(zip(np.split(X, n_batches),
                                np.split(labels, n_batches))):
        m_each_layer, v_each_layer = model.predict_each_layer(x_batch, num_samples)
        #print(m_each_layer[-2].shape)
        latent_ms[(n_b*batch_size):((n_b+1)*batch_size), :] = m_each_layer[-2][0, :, :]
    #labels = np.squeeze(labels, axis=1)
    #print(labels.shape)
    plot_tsne_no_class(latent_ms, name="dgp_htscreening", save_path=MATH_PATH)


if __name__ == '__main__':

    x_train, y_train = load_effected_action_data(path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_cleaned_ori_data.csv",
                                                 # path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_fishes_labeled.csv",
                          normalize=False, actions=[0, 1]) #, del_d=1)
    """
    x_train, y_train = load_filtered_effected_data(
        path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/0_9_25/filtered_comp_data_action.csv",
        normalize=False)
    """
    #print(x_train, y_train)
    #x_test, test_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_set.csv",
    #                      label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/train_label.csv",
    #                      normalize=False)
    #y_train = x_train
    #y_test = x_test
    #y_train, x_train, y_test, x_test, test_label = y_train[:1000, :], x_train[:1000, :], y_test[:1000, :], x_test[:1000, :], test_label[:1000, :]
    #for i in range(1000):
    #    plt.plot(x_train[i, :])
    #plt.show()
    num_inducing = 200
    Z = kmeans2(x_train, num_inducing, minit="points")[0]
    batch_size = 100
    num_samples = 1 #guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(NUMBER_LAYERS, x_train, y_train, Z)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    elbos = []
    accs = []
    for _ in range(100):
        start_time = time.time()
        elbo = training_step(dgp, x_train, y_train, batch_size)
        elbos.append(elbo)

        likelihood, acc = evaluation_step(dgp, x_train, y_train, batch_size, num_samples)
        accs.append(acc)
        duration = time.time() - start_time
        print(f"ELBO: {elbo}, Likelihood: {likelihood}, Acc: {acc} [{duration}]")
    #dgp.predict_f_compiled = tf.function(dgp.predict_each_layer,
    #                                     input_signature=[tf.TensorSpec(shape=(1, 540), dtype=tf.float64)])
    print(dgp)
    checkpoint = tf.train.Checkpoint(model=dgp)
    #save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, MATH_PATH + "models/", max_to_keep=3)
    manager.save()

    with open(os.path.join(MATH_PATH + "models/", "elbo_loss.txt"), "a+") as f:
        for i, e_l in enumerate(elbos):
            f.write("Epoch {}    elbo loss: {}    \n".format(i, e_l))

    with open(os.path.join(MATH_PATH + "models/", "ACC.txt"), "a+") as f:
        for i, ac in enumerate(accs):
            f.write("Epoch {}    acc: {}    \n".format(i, ac))
    #tf.saved_model.save(dgp, "./results/dgp_vae/dgp_vae/")
    #draw_tsne(dgp, x_test, test_label, batch_size, num_samples)