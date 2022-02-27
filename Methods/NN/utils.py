from data_loader import RAW_CLASSES, CLASSES

def plot_tsne_by_action(z_loc, classes, name, save_path="./vae_results/"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc #.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    #classes = classes.detach().cpu().numpy()
    def get_key(dict, value):
        for k, v in dict.items():
            if v == value:
                return k
        return "None"

    fig = plt.figure()
    for ic in range(4):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        ind_class = classes[:, 0] == ic
        print(ind_class)
        color = plt.cm.Set1(ic)
        action_name = get_key(CLASSES, ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, label=action_name, color=color)
        plt.title("Latent Variable T-SNE per action mode")
        fig.savefig(save_path + str(name) + "_embedding_" + str(ic) + ".png")
    fig.savefig(save_path + str(name) + "_embedding.png")