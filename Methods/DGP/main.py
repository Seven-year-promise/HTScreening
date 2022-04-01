import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from data_loader import load_cleaned_data, load_action_mode, combine_data, load_effected_data, load_effected_data_comp_action
from htscreening import make_dgp
from utils import plot_tsne, plot_latent_heatmap_by_compound, plot_tsne_no_class, \
    plot_tsne_by_compound, plot_tsne_train_eval_by_compound, \
    plot_2Ddistribution_train_eval_by_compound, plot_2Ddistribution_train_eval_by_action_mode
from feature_selection import median
from data_filter import filter_after_DGP_for_each_comp

LATENT_DIMENSION = 2
NUMBER_LAYERS = 4
MATH_PATH = "./results/dgp_vae/ae/2-dimension/"


def save_data(data, labels, action_mode={}, save_path=None):
    num = data.shape[0]
    with open(save_path, "w") as all_csv:
        all_csv_writer = csv.writer(all_csv)
        for n in range(num):
            comp_name = "C" + str(int(labels[n]))
            data_row = data[n, :].tolist()
            a_l = action_mode[labels[n]]
            all_csv_writer.writerow([comp_name]+data_row+[a_l])

def save_data_comp_action(data, labels, action_mode, save_path=None):
    num = data.shape[0]
    with open(save_path, "w") as all_csv:
        all_csv_writer = csv.writer(all_csv)
        for n in range(num):
            comp_name = "C" + str(int(labels[n]))
            data_row = data[n, :].tolist()
            a_l = str(action_mode[n])
            all_csv_writer.writerow([comp_name]+data_row+[a_l])

def save_data_action(data, action_mode=[], save_path=None):
    num = data.shape[0]
    with open(save_path, "w") as all_csv:
        all_csv_writer = csv.writer(all_csv)
        for n in range(num):
            comp_name = "C_None"
            data_row = data[n, :].tolist()
            a_l = str(action_mode[n])
            #print(a_l)
            all_csv_writer.writerow([comp_name]+data_row+[a_l])

def dgp_test_on_all(X, num_inducing=130, batch_size=100, num_samples = 1):
    Z = kmeans2(X, num_inducing, minit="points")[0]
    #num_samples = 1  # guess: for teh reparametrization, number of samples from the distribution of f_mean, f_var

    dgp = make_dgp(NUMBER_LAYERS, X, X, Z)
    checkpoint = tf.train.Checkpoint(model=dgp)
    # save_path = checkpoint.save("./results/dgp_vae/dgp_vae/")
    manager = tf.train.CheckpointManager(checkpoint, MATH_PATH + "models/", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    n_batches = max(int(len(X) / batch_size), 1)
    print(n_batches)
    latent_ms = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
    latent_vars = np.zeros((len(X), LATENT_DIMENSION), dtype=np.float)
    for n_b in range(n_batches + 1):
        x_batch = X[(n_b * batch_size):((n_b + 1) * batch_size), :]
        m_each_layer, v_each_layer = dgp.predict_each_layer(x_batch, num_samples)
        # print(m_each_layer[-2].shape)
        latent_ms[(n_b * batch_size):((n_b + 1) * batch_size), :] = m_each_layer[-1*NUMBER_LAYERS//2-1][0, :, :]
        latent_vars[(n_b * batch_size):((n_b + 1) * batch_size), :] = v_each_layer[-1*NUMBER_LAYERS//2-1][0, :, :]

    return latent_ms, latent_vars

def filter_data(ori_data, labels, latent_mu, latent_var):
    filtered_data = []
    filtered_mu = []
    filtered_labels = []
    for c in range(130):
        inds = labels == c
        comp_ori_data = ori_data[inds]
        comp_mu = latent_mu[inds]
        comp_var = latent_var[inds]
        num_comp_data = comp_ori_data.shape[0]
        #print("number of data:", num_comp_data)
        if num_comp_data < 1:
            continue

        f_ori, f_mu = filter_after_DGP_for_each_comp(comp_ori_data, comp_mu, comp_var, vote_thre=0.1, quantile=0.05)
        if len(f_ori) < 0:
            filtered_data.append(np.average(comp_ori_data, axis=0))
            filtered_mu.append(np.average(comp_mu, axis=0))
            filtered_labels.append(c)
        else:
            for f_o, f_m in zip(f_ori, f_mu):
                filtered_data.append(f_o)
                filtered_mu.append(f_m)
                filtered_labels.append(c)
    return np.array(filtered_data), np.array(filtered_mu), np.array(filtered_labels)

def filter_data_action(ori_data, labels, latent_mu, latent_var):
    filtered_data = []
    filtered_mu = []
    filtered_labels = []
    num_action = np.max(labels) + 1
    for c in range(num_action):
        inds = labels == c
        comp_ori_data = ori_data[inds]
        comp_mu = latent_mu[inds]
        comp_var = latent_var[inds]
        num_comp_data = comp_ori_data.shape[0]
        #print("number of data:", num_comp_data)
        if num_comp_data < 1:
            continue

        f_ori, f_mu = filter_after_DGP_for_each_comp(comp_ori_data, comp_mu, comp_var, vote_thre=0.5, quantile=0.25)
        if len(f_ori) < 0:
            filtered_data.append(np.average(comp_ori_data, axis=0))
            filtered_mu.append(np.average(comp_mu, axis=0))
            filtered_labels.append(c)
        else:
            for f_o, f_m in zip(f_ori, f_mu):
                filtered_data.append(f_o)
                filtered_mu.append(f_m)
                filtered_labels.append(c)
    return np.array(filtered_data), np.array(filtered_mu), np.array(filtered_labels)

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
    x_train, train_label, actions = load_effected_data_comp_action(path="/srv/yanke/PycharmProjects/HTScreening/data/raw_data/effected_compounds_pvalue_frames_labeled.csv",
                          normalize=False)
    '''
    x_eval, eval_label = load_cleaned_data(path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_set.csv",
                          label_path="/srv/yanke/PycharmProjects/HTScreening/data/cleaned_dataset/eval_label.csv",
                          normalize=False)
    
    all_data, all_labels = combine_data(x_train, x_eval, train_label, eval_label)
    
    action_dict = load_action_mode(path="/srv/yanke/PycharmProjects/HTScreening/data/data_median_all_label.csv")
    '''
    save_data_comp_action(x_train, train_label.reshape(-1, ), actions.reshape(-1, ),
              save_path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/original_comp_data_action.csv")

    latent_mu, latent_var = dgp_test_on_all(x_train, num_inducing=130, batch_size=100, num_samples = 1)

    save_data_comp_action(latent_mu, train_label.reshape(-1, ), actions.reshape(-1, ),
              save_path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/original_comp_mu_action2.csv")
    """
    ori_after_filter, latent_mu_after_filter, labels_after_filter = filter_data_action(x_train, train_label.reshape(-1, ), latent_mu, latent_var)
    save_data_action(ori_after_filter, labels_after_filter,
              save_path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/filtered_comp_data_action.csv")
    save_data_action(latent_mu_after_filter, labels_after_filter,
              save_path="/srv/yanke/PycharmProjects/HTScreening/Methods/DGP/results/dgp_vae/effected/saved_data/2d/filtered_comp_mu_action.csv")
    """