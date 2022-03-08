import numpy as np
from scipy.stats import multivariate_normal as mvn

def filter_after_DGP_for_each_comp(data, latent_mu, latent_var, vote_thre=0.5, quantile=0.05):
    """
    data: the original data
    latent: the latent space of the data
    vote_thre: the percentage of the data number
    quantile:
    gaussian distribution eclipse
    ((x-mu_x) / sigma_x)**2 + ((y-mu_y) / sigma_y)**2 = s
    s = 5.991, 95%
    s = 4.605, 90%
    """
    # TODO
    data_after_vote = []
    latent_after_vote = []
    num = data.shape[0]
    vote_num_thre = int(vote_thre * num)
    for n in range(num):
        #print("data: ", n)
        ori_d = data[n, :]
        l_mu = latent_mu[n, :]
        l_var = latent_var[n, :]**2
        dist = mvn(mean=l_mu, cov=np.diag(l_var))
        votes = 0
        for l_n in range(num):
            l_ele = latent_mu[l_n, :]
            prob = dist.cdf(l_ele)
            #print(l_mu, l_ele, prob)
            if prob > quantile and prob < (1-quantile):
                votes += 1
        #print(votes)
        if votes > vote_num_thre:
            data_after_vote.append(ori_d)
            latent_after_vote.append(l_mu)
    return data_after_vote, latent_after_vote



if __name__ == "__main__":
    data = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])
    mean = np.array([[3.4, 5.6],
                     [6.5, 5.4],
                     [23.3, 15.7]])
    var = np.array([[0.4, 0.6],
                     [0.5, 0.04],
                     [0.13, 0.07]])
    filter_after_DGP_for_each_comp(data, mean, var, 0.9)