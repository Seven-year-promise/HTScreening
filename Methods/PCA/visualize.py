import pickle
import matplotlib.pyplot as plt


im = pickle.load(open("./results/pca_with_name/train/VAE_embedding.pickle", "rb"))
plt.show()