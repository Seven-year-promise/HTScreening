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
from sklearn.metrics import confusion_matrix
import pandas as pd

LATENT_DIMENSION = 10
MATH_PATH = "./results/dgp_vae/effected/class/22-feature/binary/"
NUMBER_LAYERS = 4

def make_dgp(num_layers, X, Y, Z):
    kernels = [RBF(variance=2.0, lengthscales=2.0)]
    layer_sizes = [22, 10, 10, LATENT_DIMENSION]
    for l in range(num_layers-1):
        kernels.append(RBF(variance=2.0, lengthscales=2.0))
    model = DeepGP(X, Y, Z, kernels, layer_sizes, MultiClass(2),
                   num_outputs=2)

    # init hidden layers to be near deterministic
    for layer in model.layers[:-1]:
        layer.q_sqrt.assign(layer.q_sqrt * 1e-5)
    return model

def sample_ratio_control(X, Y, batch_size):

    class_data0 = X[Y[:, 0]==0, :]
    class_label0 = Y[Y[:, 0]==0, :]
    patch_choice0 = np.random.choice(len(class_label0), batch_size//2)
    class_data1 = X[Y[:, 0]>0, :]
    class_label1 = Y[Y[:, 0]>0, :]
    patch_choice1 = np.random.choice(len(class_label1), batch_size // 2)

    X_batch = np.zeros((batch_size, X.shape[1]), np.float)
    Y_batch = np.zeros((batch_size, Y.shape[1]), np.float)

    X_batch[:batch_size//2] = class_data0[patch_choice0, :]
    Y_batch[:batch_size // 2] = class_label0[patch_choice0, :]

    X_batch[batch_size // 2:] = class_data1[patch_choice1, :]
    Y_batch[batch_size // 2:] = class_label1[patch_choice1, :]

    patch_choice = np.random.choice(batch_size, batch_size)

    return X_batch[patch_choice, :], Y_batch[patch_choice, :]



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
        #x_batch, y_batch = sample_ratio_control(X, Y, batch_size)
        #print(x_batch, y_batch)
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
    #confusion_m = np.zeros((3,3))

    y_actual = []
    y_pred = []
    for n_b in range(n_batches + 1):
        if n_b == n_batches:
            x_batch = X[(n_b * batch_size):, :]
            y_batch = Y[(n_b * batch_size):, :]
            m, v = model.predict_y(x_batch, num_samples)
            likelihood[(n_b * batch_size):] = model.predict_log_density((x_batch, y_batch), num_samples)
            acc[(n_b * batch_size):, :] = (mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
            y_actual += y_batch.astype(int).tolist()
            y_pred += mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int).tolist()
        else:
            x_batch = X[(n_b * batch_size):((n_b + 1) * batch_size), :]
            y_batch = Y[(n_b * batch_size):((n_b + 1) * batch_size), :]
            m, v = model.predict_y(x_batch, num_samples)
            likelihood[(n_b * batch_size):((n_b + 1) * batch_size)] = model.predict_log_density((x_batch, y_batch), num_samples)
            acc[(n_b * batch_size):((n_b + 1) * batch_size), :] = (
                        mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int) == y_batch.astype(int))
            y_actual += y_batch.astype(int).tolist()
            y_pred += mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int).tolist()
        #print(m)
        #confusion_m += confusion_matrix(y_batch.astype(int), mode(np.argmax(m, 2), 0)[0].reshape(y_batch.shape).astype(int))
    #print(confusion_m)

    y_actu = pd.Series(np.array(y_actual).reshape(-1,), name='Actual')
    y_pred = pd.Series(np.array(y_pred).reshape(-1,), name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
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

    x_train, y_train = load_effected_action_data(path="/srv/yanke/PycharmProjects/HTScreening/data/effected/effected_feature_train_set.csv",
                                                 # path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_fishes_labeled.csv",
                          normalize=False, actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) #, del_d=1)
    x_eval, y_eval = load_effected_action_data(
        path="/srv/yanke/PycharmProjects/HTScreening/data/effected/effected_feature_eval_set.csv",
        # path="/srv/yanke/PycharmProjects/HTScreening/data/effected_compounds_fishes_labeled.csv",
        normalize=False, actions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # , del_d=1)
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

        likelihood, acc = evaluation_step(dgp, x_eval, y_eval, batch_size, num_samples)
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