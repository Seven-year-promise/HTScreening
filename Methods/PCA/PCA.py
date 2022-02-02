"""
Principal Component Analysis for the normalized data
by Yanke
"""
import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import io


class PrincipalComponentAnalysis():
    def __init__(self, n_components=2, svd_solver='full'):
        self.pca = PCA(n_components, svd_solver)

    def fit(self, data):
        self.pca.fit(data)

    def get_singular_value(self, data):
        U, s, Vh = linalg.svd(data)
        return U, s, Vh


class PCA_torch:

    def __init__(self, center=False, n_components=2):
        self.n_components = n_components
        self.is_center = center
        self.eigenpair = []
        self.data = []

    def Center(self, Data):
        # Convert to torch Tensor and keep the number of rows and columns
        t = torch.from_numpy(Data)
        no_rows, no_columns = t.size()
        row_means = torch.mean(t, 1)
        # Expand the matrix in order to have the same shape as X and substract, to center
        for_subtraction = row_means.expand(no_rows, no_columns)
        X = t - for_subtraction  # centered
        return (X)

    def SingularValue(self, Data):
        # Center the Data using the static method within the class
        if self.center:
            X = self.Center(Data)
        else:
            X = torch.from_numpy(Data)  # cls.Center(Data)
        U, S, V = torch.svd(X)
        self.data = X
        return U, S, V

    def fit_SVD(self, Data):
        U, S, V = self.SingularValue(Data)
        eigvecs = U.t()[:, :self.n_components]  # the first k vectors will be kept
        print(U.size(), eigvecs.size())
        y = torch.mm(U, eigvecs)

        # Save variables to the class object, the eigenpair and the centered data
        self.eigenpair = (eigvecs, S)

        return (y)

    def test(self, data):
        return np.dot(data, self.eigenpair[0])

    def fit_PCA(self, Data):
        X = Data
        n, m = X.shape

        # mu = X.mean(axis=0)
        # X = X - mu
        # Compute covariance matrix
        C = np.dot(X.T, X)  # / (n - 1)
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(C)
        # print(eigen_vecs)
        io.savemat('eigen_vec.mat', mdict={'eigen_vec': eigen_vecs})
        # np.savetxt('text.txt', eigen_vecs.real, fmt='%.2f')
        # Project X onto PC space
        eigen_vals = eigen_vals[:self.n_components]
        eigen_vecs = eigen_vecs[:, :self.n_components]
        self.eigenpair = (eigen_vecs, eigen_vals)
        X_pca = np.dot(X, eigen_vecs)

        """
        n, m = Data.shape
        if self.is_center:
            #X = self.Center(Data)
            pass
        else:
            X = torch.from_numpy(Data)  # cls.Center(Data)
        X_T = torch.transpose(X, 0, 1)
        print(X_T.size(), X.size())
        C = torch.mm(X_T, X)
        eigen_vals, eigen_vecs = torch.eig(C)
        print(eigen_vecs)

        X_pca = torch.mm(X, eigen_vecs)
        """
        return (X_pca)

    def explained_variance(self):
        # Total sum of eigenvalues (total variance explained)
        tot = sum(self.eigenpair[1])
        # Variance explained by each principal component
        var_exp = [(i / tot) for i in sorted(self.eigenpair[1], reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        # X is the centered data
        X = self.data
        # Plot both the individual variance explained and the cumulative:
        plt.bar(range(X.size()[1]), var_exp, alpha=0.5, align='center', label='individual explained variance')
        plt.step(range(X.size()[1]), cum_var_exp, where='mid', label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.show()