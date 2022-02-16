import torch.nn as nn
import torch

criterion = nn.BCELoss(reduction='sum')
MSE = nn.MSELoss(reduction="sum")
cross_entropy = nn.CrossEntropyLoss(reduction="sum")
def AeLoss(recon_x,x):
    MSE_loss = MSE(recon_x, x)#/10000000
    loss = MSE_loss
    return loss

def VaeLoss(recon_x,x,mu,logvar):
    MSE_loss = MSE(recon_x, x)#/10000000
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# * 100.
    loss = MSE_loss+KLD
    return loss, MSE_loss, KLD