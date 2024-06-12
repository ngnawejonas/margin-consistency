import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from .gairat import GAIR, adjust_Lambda, weight_assign

def calc_entropy(x, from_logits=True, dim=1):
    if from_logits:
      log_probs = x.log_softmax(dim=-1)
    else:
      log_probs = torch.log(x+1e-12)

    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    predictive_entropy = -p_log_p.sum(axis=dim)
    return predictive_entropy

def ce_loss(model,
            x_natural,
            y,
            **args):
    model.train()
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    return loss_natural

# https://github.com/YisenWang/MART/blob/master/mart.py 
# & # https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training/
def generate_adv(model,
                x_natural,
                y=None,
                step_size=None,
                epsilon=None,
                perturb_steps=None,
                norm='Linf',
                category='madry',
                rand_init=True,
                gairat=False,
                gpu_id=None):
    if y is None:
        outputs = model(x_natural)
        y = torch.max(outputs.data, 1)[1]
    batch_size = len(x_natural)
    if 'trades' in category  and rand_init:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=gpu_id).detach() 
    if 'madry' in category and rand_init:
        epsilonx = 1
        x_adv = x_natural.detach() +  torch.zeros_like(x_natural, dtype=torch.float, device=gpu_id).uniform_(-epsilonx, epsilonx).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if not rand_init:
        x_adv = x_natural.detach()

    if gairat:
        Kappa = torch.zeros(len(x_natural))

    if norm == 'Linf':
        for _ in range(perturb_steps):
            if gairat:
                output = model(x_adv)
                predict = output.max(1, keepdim=True)[1]
                # Update Kappa
                for p in range(len(x_adv)):
                    if predict[p] == y[p]:
                        Kappa[p] += 1            
            x_adv = x_adv.clone().detach().to(torch.float).requires_grad_()
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if 'trades' in category:
                    logit_adv = model(x_adv)
                    logit = model(x_natural)
                    if category == 'htrades':
                        criterion_mse = nn.MSELoss(reduction='sum').to(gpu_id)
                        loss_adv = criterion_mse(calc_entropy(logit_adv, dim=1), calc_entropy(logit, dim=1))
                    else:
                        criterion_kl = nn.KLDivLoss(reduction='sum').to(gpu_id)
                        loss_adv = criterion_kl(F.log_softmax(logit_adv, dim=1),F.softmax(logit, dim=1))
                else:
                    loss_adv = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_adv, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == 'L2':
        delta = 0.001 * torch.randn(x_natural.shape).to(gpu_id).detach()
        delta.requires_grad_()

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            x_adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                if 'trades' not in category:
                    loss_attack = (-1)*F.cross_entropy(model(x_adv), y)
                else:
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    log_logit = F.log_softmax(model(x_adv), dim=1)
                    logit = F.softmax(model(x_natural), dim=1)
                    loss_attack = (-1)*criterion_kl(log_logit, logit) 
            loss_attack.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = x_natural + delta
        x_adv.requires_grad=False
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv.requires_grad = False
    if gairat:
        return x_adv, Kappa
    return x_adv

#https://github.com/yaodongyu/TRADES/blob/master/trades.py
def trades_loss(model,
                x_natural: Tensor,
                y,
                gpu_id=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                bias=-1.5, slope=1.0,
                pmargin=False,
                norm='Linf',
                **args):
    model.eval()
    batch_size = len(x_natural)
    # # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,
                y=None,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                category='trades',
                norm=norm,
                gpu_id=gpu_id)
    model.train()
    # zero gradient
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)
    if pmargin:
        kl = nn.KLDivLoss(reduction='none').to(gpu_id)
        norm_weight = weight_assign(logits_adv, y, bias, slope, gpu_id=gpu_id)
        loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * norm_weight)
    else:
        kl = nn.KLDivLoss(reduction="sum").to(gpu_id)
        loss_robust = (1.0 / batch_size) * kl(torch.log(adv_probs + 1e-12), nat_probs)

    loss_natural = F.cross_entropy(logits, y)
    loss =  loss_natural + beta * loss_robust
    return loss

def pm_trades_loss(model, x_natural, y, gpu_id=None, **args):
    return trades_loss(model, x_natural, y,  pmargin=True, gpu_id=gpu_id, **args)

def pm_madry_loss(model, x_natural,y, gpu_id=None, **args):
    return madry_loss(model, x_natural, y, pmargin=True, gpu_id=gpu_id, **args)

#@title madry et al. loss
def madry_loss(model,
                x_natural,
                y,
                gpu_id=None, 
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                pmargin=False,
                bias=-1.5, slope=1.0,
                norm='Linf',
                **args):
    model.eval()
    # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,
                y,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                norm=norm,
                gpu_id=gpu_id) 
    model.train()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # zero the parameter gradients
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits_adv = model(x_adv)
    if pmargin:
        norm_weight = weight_assign(logits_adv, y, bias, slope, gpu_id=gpu_id)
        loss = F.cross_entropy(logits_adv, y, reduction = 'none')
        loss = (loss * norm_weight).mean()
    else:
        loss = F.cross_entropy(logits_adv, y)
    return loss

#based on table 1 in https://openreview.net/pdf?id=rklOg6EFwS
def alp_loss(model,
                x_natural,
                y,
                gpu_id=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                norm='Linf', **args):
    model.eval()
    batch_size = len(x_natural)
    # # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,
                y,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                norm=norm,
                gpu_id=gpu_id)
    model.train()
    # zero gradient
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)
    
    loss_adv = F.cross_entropy(logits_adv, y)
    mse = nn.MSELoss(reduction="sum")
    loss_robust = (1.0 / batch_size) * mse(adv_probs, nat_probs)
    loss = loss_adv + beta * loss_robust
    return loss

#based on table 1 in https://openreview.net/pdf?id=rklOg6EFwS
def clp_loss(model,
                x_natural,
                y,
                gpu_id=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                norm='Linf', **args):
    model.eval()
    batch_size = len(x_natural)
    # # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,
                y,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                norm=norm,
                gpu_id=gpu_id)    
    model.train()    
    # zero gradient
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)

    loss_natural = F.cross_entropy(logits, y)
    mse = nn.MSELoss(reduction="sum")
    loss_robust = (1.0 / batch_size) * mse(adv_probs, nat_probs)
    loss = loss_natural + beta * loss_robust
    return loss

#based on table 1 in https://openreview.net/pdf?id=rklOg6EFwS
def mma_loss(model,
                x_natural,
                y,
                gpu_id=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                norm='Linf', **args):
    model.eval()
    batch_size = len(x_natural)
    # # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,y,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                norm=norm,
                gpu_id=gpu_id)
    model.train()
    # zero gradient
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    # print(logits.sum())
    clean_loss = F.cross_entropy(logits, y, reduction='none')
    adv_loss = F.cross_entropy(logits_adv, y, reduction='none')
    _, predicted = torch.max(logits.data, 1)
    misclassified = (predicted!=y)
    correct = (predicted==y)
    loss = (1.0 / batch_size) * (torch.sum(clean_loss*misclassified) + torch.sum(adv_loss*correct))
    return loss

# https://github.com/YisenWang/MART/blob/master/mart.py
def mart_loss(model,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              bias=-1.5, slope=1.0,
              norm='Linf',
              pmargin=False,
              gpu_id=None,
              **args):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = generate_adv(model,x_natural,y,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                norm=norm,
                gpu_id=gpu_id)

    model.train()

    # zero gradient
    model.zero_grad(set_to_none=True)

    logits = model(x_natural)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    if pmargin:
        loss_adv = F.cross_entropy(logits_adv, y, reduction = 'none') + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y, reduction = 'none')
        norm_weight = weight_assign(logits_adv, y, bias, slope, gpu_id=gpu_id)
        loss_adv = (loss_adv * norm_weight).mean()
    else:
        loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    
    loss = loss_adv + beta * loss_robust
    return loss

def htrades_loss(model,
                x_natural: Tensor,
                y,
                gpu_id=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                bias=-1.5, slope=1.0,
                pmargin=False,
                norm='Linf',
                **args):
    model.eval()
    batch_size = len(x_natural)
    # # generate adversarial example
    x_adv = generate_adv(model,
                x_natural,
                y=None,
                step_size=step_size,
                epsilon=epsilon,
                perturb_steps=perturb_steps,
                category='trades',
                norm=norm,
                gpu_id=gpu_id)
    model.train()
    # zero gradient
    model.zero_grad(set_to_none=True)
    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)
    if pmargin:
        mse = nn.MSELoss(reduction='none').to(gpu_id)
        norm_weight = weight_assign(logits_adv, y, bias, slope, gpu_id=gpu_id)
        loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(mse(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * norm_weight)
    else:
        mse = nn.MSELoss(reduction="sum").to(gpu_id)
        loss_robust = (1.0 / batch_size) * mse(torch.log(adv_probs + 1e-12), nat_probs)

    loss_natural = F.cross_entropy(logits, y)
    loss =  loss_natural + beta * loss_robust
    return loss

# https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training
def gairat_loss(model,
              x_natural,
              y,
              epoch=None,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              norm='Linf',
              num_epochs=100,
              weight_assignment_function='Tanh',
              begin_epoch=30,
              gpu_id=None, **args):
    model.eval()
    # batch_size = len(x_natural)
    # Get adversarial data and geometry value
    x_adv, Kappa = generate_adv(model,
                                x_natural,
                                y,
                                step_size=step_size,
                                epsilon=epsilon,
                                perturb_steps=perturb_steps,
                                norm=norm,
                                category="madry",
                                rand_init=True,
                                gairat=True,
                                gpu_id=gpu_id)

    model.train()
    model.zero_grad(set_to_none=True)
    adv_logits = model(x_adv)
    if (epoch + 1) >= begin_epoch:
        Kappa = Kappa.to(gpu_id)
        # Get lambda
        Lambda = adjust_Lambda(epoch + 1,num_epochs)
        loss = nn.CrossEntropyLoss(reduction='none')(adv_logits, y)
        # Calculate weight assignment according to geometry value
        normalized_reweight = GAIR(perturb_steps, Kappa, Lambda, weight_assignment_function)
        loss = loss.mul(normalized_reweight).mean()
    else:
        loss = F.cross_entropy(adv_logits, y)

    # loss = loss * batch_size
    return loss
