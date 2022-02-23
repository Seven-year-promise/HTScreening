import gpflow
import tensorflow as tf
from gpflow.likelihoods import Likelihood, Gaussian
import numpy as np
from data_loader import RAW_CLASSES

class BroadcastingLikelihood(Likelihood):
    """
    A wrapper for the likelihood to broadcast over the samples dimension.
    The Gaussian doesn't need this, but for the others we can apply reshaping
    and tiling. With this wrapper all likelihood functions behave correctly
    with inputs of shape S,N,D, but with Y still of shape N,D
    """

    def __init__(self, likelihood):
        super().__init__(likelihood.latent_dim, likelihood.observation_dim)
        self.likelihood = likelihood

        if isinstance(likelihood, Gaussian):
            self.needs_broadcasting = False
        else:
            self.needs_broadcasting = True

    def _broadcast(self, f, vars_SND, vars_ND):
        if not self.needs_broadcasting:
            return f(vars_SND, [tf.expand_dims(v, 0) for v in vars_ND])
        else:
            S, N, D = [tf.shape(vars_SND[0])[i] for i in range(3)]
            vars_tiled = [tf.tile(x[None, :, :], [S, 1, 1]) for x in vars_ND]

            flattened_SND = [tf.reshape(x, [S*N, D]) for x in vars_SND]
            flattened_tiled = [tf.reshape(x, [S*N, -1]) for x in vars_tiled]

            flattened_result = f(flattened_SND, flattened_tiled)
            if isinstance(flattened_result, tuple):
                return [tf.reshape(x, [S, N, -1]) for x in flattened_result]
            else:
                return tf.reshape(flattened_result, [S, N])

    def _variational_expectations(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.variational_expectations(vars_SND[0],
                                                                               vars_SND[1],
                                                                               vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])

    def _log_prob(self, F, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.log_prob(vars_SND[0],
                                                               vars_ND[0])
        return self._broadcast(f, [F], [Y])

    def _conditional_mean(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_mean(
            vars_SND[0])
        return self._broadcast(f, [F], [])

    def _conditional_variance(self, F):
        f = lambda vars_SND, vars_ND: self.likelihood.conditional_variance(
            vars_SND[0])
        return self._broadcast(f, [F], [])

    def _predict_mean_and_var(self, Fmu, Fvar):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_mean_and_var(
            vars_SND[0],
            vars_SND[1])
        return self._broadcast(f, [Fmu, Fvar], [])

    def _predict_log_density(self, Fmu, Fvar, Y):
        f = lambda vars_SND, vars_ND: self.likelihood.predict_log_density(
            vars_SND[0],
            vars_SND[1],
            vars_ND[0])
        return self._broadcast(f, [Fmu, Fvar], [Y])


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + gpflow.default_jitter()) ** 0.5
    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = gpflow.default_jitter() * tf.eye(N, dtype=gpflow.default_float())[None, None, :, :] # 11NN
        chol = tf.linalg.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND


def summarize_tensor(x, title=""):
    print("-"*10, title, "-"*10, sep="")
    shape = x.shape
    print(f"Shape: {shape}")

    nans = tf.reduce_sum(tf.cast(tf.math.is_nan(x), tf.int16))
    print(f"NaNs: {nans}")

    nnz = tf.reduce_sum(tf.cast(x < 1e-8, tf.int16))
    print(f"NNZ: {nnz}")

    mean = tf.reduce_mean(x)
    print(f"Mean: {mean}")
    std = tf.math.reduce_std(x)
    print(f"Std: {std}")

    min = tf.math.reduce_min(x)
    print(f"Min: {min}")
    max = tf.math.reduce_max(x)
    print(f"Max: {max}")
    print("-"*(20+len(title)))



def plot_tsne(z_loc, classes, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    #classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(ic) + ".png")
    fig.savefig(save_path + str(name) + "_embedding.png")

def plot_tsne_no_class(z_loc, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    #classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    color = plt.cm.Set1(0)
    plt.scatter(z_embed[:, 0], z_embed[:, 1], s=10, color=color)
    plt.title("Latent Variable T-SNE per Class")
    fig.savefig(save_path + str(name) + "_embedding.png")

def plot_latent_heatmap_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    """
    This is used to generate a distribution of the samples
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_data = []
    for c in range(100, 120):
        inds = labels == c
        comp_data = z_loc[inds]
        num_comp_data = comp_data.shape[0]
        if num_comp_data < 1:
            continue
        for c_n in range(num_comp_data):
            display_data.append(comp_data[c_n, :])
        display_data.append(np.ones(comp_data.shape[1], dtype=np.float)*(-10))

    fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    #heatmap = axis.pcolor(display_data, cmap=plt.cm.Blues)  # heatmap contient les valeurs
    im = plt.imshow(np.array(display_data), cmap="cool", interpolation="nearest")
    plt.colorbar(im)

    fig.set_size_inches(11.03, 3.5)
    plt.title("Latent of all data")
    plt.savefig(save_path  + "latent_embedding_all" + ".png")
    plt.clf()

def plot_tsne_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    #classes = classes.detach().cpu().numpy()
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    for c in range(130):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        inds = labels == c
        comp_data = z_embed[inds]
        num_comp_data = comp_data.shape[0]
        if num_comp_data < 1:
            continue
        fig = plt.figure()

        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=10, color=color0, alpha=0.05)
        plt.scatter(comp_data[:, 0], comp_data[:, 1], s=10, color=color1)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(c) + ".png")
        plt.clf()

def plot_tsne_train_eval_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    color2 = plt.cm.Set1(3)
    for c in range(130):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = z_embed_all["train"][train_inds]
        eval_comp_data = z_embed_all["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue
        fig = plt.figure()

        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.05, label="all")
        plt.scatter(train_comp_data[:, 0], train_comp_data[:, 1], s=5, color=color1, label = "train C" + str(c))
        plt.scatter(np.average(train_comp_data,axis=0)[0], np.average(train_comp_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_comp_data[:, 0], eval_comp_data[:, 1], s=5, color=color2, label="eval C" + str(c))
        plt.scatter(np.average(eval_comp_data, axis=0)[0], np.average(eval_comp_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        plt.legend(loc="best")
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(c) + ".png")
        plt.clf()


def plot_2Ddistribution_train_eval_by_compound(z_loc, labels, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    #model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = z_states # model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    color0 = plt.cm.Set1(0)
    color1 = plt.cm.Set1(1)
    color2 = plt.cm.Set1(3)
    for c in range(130):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        train_inds = labels["train"] == c
        eval_inds = labels["eval"] == c
        train_comp_data = z_embed_all["train"][train_inds]
        eval_comp_data = z_embed_all["eval"][eval_inds]
        num_train_comp_data = train_comp_data.shape[0]
        num_eval_comp_data = eval_comp_data.shape[0]
        if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
            continue
        fig = plt.figure()

        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.05, label="all")
        plt.scatter(train_comp_data[:, 0], train_comp_data[:, 1], s=5, color=color1, label = "train C" + str(c))
        plt.scatter(np.average(train_comp_data,axis=0)[0], np.average(train_comp_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_comp_data[:, 0], eval_comp_data[:, 1], s=5, color=color2, label="eval C" + str(c))
        plt.scatter(np.average(eval_comp_data, axis=0)[0], np.average(eval_comp_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        #plt.xlim(-1.45, -1.3)
        #plt.ylim(-0.7, -0.55)
        plt.legend(loc="best")
        plt.title("Latent Variable first 2D per Class")
        fig.savefig(save_path + str(name) + "_embedding_" + str(c) + ".png")
        plt.clf()

def plot_2Ddistribution_train_eval_by_action_mode(z_loc, labels, action_modes, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    train_num = len(z_loc["train"])
    eval_num = len(z_loc["eval"])
    all_data = np.zeros((train_num+eval_num, z_loc["train"].shape[1]), np.float)
    all_data[:train_num, :] = z_loc["train"]
    all_data[train_num:, :] = z_loc["eval"]
    #model_tsne = TSNE(n_components=2, random_state=0)
    z_states = all_data #.detach().cpu().numpy()
    z_embed = z_states # model_tsne.fit_transform(z_states)
    z_embed_all = {}
    z_embed_all["train"] = z_embed[:train_num, :]
    z_embed_all["eval"] = z_embed[train_num:, :]
    #classes = classes.detach().cpu().numpy()
    def get_key(dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return "None"
    colors = ["tab:blue", "tab:gray", "tab:pink", "tab:red", "tab:green",
              "tab:purple", "tab:orange", "tab:cyan", "tab:olive", "tab:brown"]

    for a_name, a_num in RAW_CLASSES.items():
        print(a_name, a_num)
        fig = plt.figure()
        color0 = plt.cm.Set1(0)
        color1 = plt.cm.Set1(1)
        color2 = plt.cm.Set1(3)
        plt.scatter(z_embed[:, 0], z_embed[:, 1], s=5, color=color0, alpha=0.25, label="all")
        train_action_data = []
        eval_action_data = []
        for c in range(130):
            train_inds = labels["train"] == c
            eval_inds = labels["eval"] == c
            train_comp_data = z_embed_all["train"][train_inds]
            eval_comp_data = z_embed_all["eval"][eval_inds]
            num_train_comp_data = train_comp_data.shape[0]
            num_eval_comp_data = eval_comp_data.shape[0]
            if (num_train_comp_data < 1) and (num_eval_comp_data < 1):
                continue
            if action_modes[c] != a_num:
                continue
            print(c, action_modes[c], a_num)

            for n_t_c in range(num_train_comp_data):
                train_action_data.append(train_comp_data[n_t_c, :])
            for n_e_c in range(num_eval_comp_data):
                eval_action_data.append(eval_comp_data[n_e_c, :])
        train_action_data = np.array(train_action_data)
        eval_action_data = np.array(eval_action_data)

        plt.scatter(train_action_data[:, 0], train_action_data[:, 1], s=5, color=color1, label = a_name)
        plt.scatter(np.average(train_action_data,axis=0)[0], np.average(train_action_data,axis=0)[1], s=100, alpha=0.8, color=color1)
        plt.scatter(eval_action_data[:, 0], eval_action_data[:, 1], s=5, color=color2, label=a_name)
        plt.scatter(np.average(eval_action_data, axis=0)[0], np.average(eval_action_data, axis=0)[1], s=100, alpha=0.8, color=color2)
        #plt.xlim(-1.45, -1.3)
        #plt.ylim(-0.7, -0.55)
        plt.legend(loc="best")
        plt.title("Latent Variable first 2D per action mode")
        fig.savefig(save_path + str(name) + "_embedding_action" + str(a_num) + ".png")
        plt.clf()