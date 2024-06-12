import torch
import torch.nn.functional as F
import numpy as np

# https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training/
def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        hsteps = int(num_steps/2)
        reweight = ((Lambda+(hsteps-Kappa)*5/(hsteps)).tanh() +1 )/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight


# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch,num_epochs, Lambda=1.0,
                   Lambda_max=float('inf'),
                   Lambda_schedule='fixed'):
    Lam = float(Lambda)
    # Train ResNet
    Lambda = Lambda_max
    if Lambda_schedule == 'linear':
        if epoch >= 30:
            Lambda = Lambda_max - (epoch/num_epochs) * (Lambda_max - Lam)
    elif Lambda_schedule == 'piecewise':
        if epoch >= 30:
            Lambda = Lam
        elif epoch >= 60:
            Lambda = Lam-2.0
    elif Lambda_schedule == 'fixed':
        if epoch >= 30:
            Lambda = Lam
    return Lambda


# https://github.com/QizhouWang/MAIL/blob/main/mail_loss.py
def PM(logit, target, gpu_id=None):
    eye = torch.eye(10, device=gpu_id)
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()
    top2_probs = logit.softmax(1).topk(2, largest = True)
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    return  probs_2nd - probs_GT

def weight_assign(logit, target, bias=-1.5, slope=1.0, gpu_id=None):
    pm = PM(logit, target, gpu_id=gpu_id)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    k = len(reweight)/reweight.sum() # 3
    normalized_reweight = reweight * k
    return normalized_reweight