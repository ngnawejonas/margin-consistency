import torch
import numpy as np
from torchmetrics.regression import KLDivergence
from sklearn.metrics import det_curve
from tqdm import tqdm

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    return torch.div(logits, temperature)

def get_probs(logits, temp=1.0, verbose=True):
    sm = torch.nn.Softmax(dim=1)
    probs = sm(temperature_scale(logits, temp))
    return probs.cpu().detach()

# entropy
def calc_entropy(x, from_logits=True):
    if from_logits:
        log_probs = torch.nn.LogSoftmax(dim=1)(x+1e-8)
    else:
        log_probs = torch.log(x)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    predictive_entropy = -p_log_p.sum(axis=1)
    return predictive_entropy

def calc_qscore(x, from_logits=True):
    if from_logits:
        probs = get_probs(x)
        qscore = 1 - (probs**2).sum(axis=1)
    else:
        qscore = 1 - (x**2).sum(axis=1)
    return qscore

def calc_maxprob(x, from_logits=True):
    if from_logits:
        probs = get_probs(x)
        pm =  probs.max(dim=1).values
    else:
        pm =  x.max(dim=1).values
    return pm

def calc_maxlogits(x):
    return x.max(1).values

def calc_pmarginadv(out, out_adv, from_logits=True):
    if from_logits:
        probs = get_probs(out)
        probsadv = get_probs(out_adv)
        pm =  probs.max(dim=1).values
        pmadv = probsadv.max(dim=1).values
    else:
        pm =  out.max(dim=1).values
        pmadv = out_adv.max(dim=1).values
    return pm - pmadv
def calc_lmarginadv(out, out_adv, from_logits=True):
    return calc_pmarginadv(out, out_adv, from_logits=False)

def calc_pmargin(x, from_logits=True, **args):
    if from_logits:
        probs = get_probs(x)
        pm =  probs.sort(dim=1, descending=True).values
    else:
        pm =  x.sort(dim=1, descending=True).values
    return pm[:,0] - pm[:,1]

def calc_lmargin(x, **args):
    return calc_pmargin(x, from_logits=False, **args)

def calc_tpmargin(x, y, from_logits=False):
    if from_logits:
        probs = get_probs(x)
        true_probs, non_true_probs = extract_non_true_probs(probs, y)
        pm =  true_probs - non_true_probs.max(1)[0]
    else:
        true_probs, non_true_probs = extract_non_true_probs(x, y)
        pm =  true_probs - non_true_probs.max(1)[0]
    return pm

def calc_tcp(x, y, from_logits=False):
    if from_logits:
        return torch.gather(get_probs(x), dim=1, index=y.reshape(-1,1)).flatten()
    else:
        return torch.gather(x, dim=1, index=y.reshape(-1,1)).flatten()  

def calc_sphere(x, alpha=2, from_logits=True):
    if from_logits:
        probs = get_probs(x)
        sphscore = (probs**alpha).sum(axis=1)
        sphscore = sphscore**(1/alpha)
    else:
        sphscore = (x**alpha).sum(axis=1)
        sphscore = sphscore**(1/alpha)
    return sphscore

def doctor_score1(x, from_logits=True):
    if from_logits:
        probs = get_probs(x)
        g = (probs**2).sum(axis=1)
    else:
        g = (x**2).sum(axis=1)
    docscore = (1 - g)/g
    return docscore

def doctor_score2(x, from_logits=True):
    if from_logits:
        probs = get_probs(x)
        pm =  probs.max(dim=1).values
    else:
        pm =  x.max(dim=1).values
    docscore = (1 - pm)/pm
    return docscore


def calc_kldivx(out, out_adv, from_logits=True):
    kldiv = KLDivergence(reduction=None)

    if from_logits:
        pout = get_probs(out)
        pout_adv = get_probs(out_adv)
        return 0.5*(kldiv(pout, pout_adv)+kldiv(pout_adv, pout))

    return 0.5*(kldiv(out, out_adv)+kldiv(out_adv, out))

def calc_drep(out, out_adv, norm=float('inf')):
    return compute_norm(out-out_adv, norm)

def calc_kldiv(out, out_adv, from_logits=True):
    kldiv = KLDivergence(reduction=None)

    if from_logits:
        pout = get_probs(out)
        pout_adv = get_probs(out_adv)
        return kldiv(pout_adv, pout)

    return kldiv(pout_adv, out)

def calc_energy(logits, T=1.):
    return -T*torch.logsumexp(logits/T, dim=1)

def compute_norm(x, norm):
  # with torch.no_grad():
  return torch.linalg.norm(x.view(x.shape[0], -1), dim=1,  ord=norm)


def extract_non_true_probs(probs, true_targets):
    """
    Extract probabilities for non-true targets from a tensor of probabilities.

    Parameters:
    - probs (torch.Tensor): Tensor of predicted probabilities with shape (batch_size, num_classes).
    - true_targets (torch.Tensor): Tensor of true class labels with shape (batch_size,).

    Returns:
    - true_probs (torch.Tensor): Tensor of predicted probabilities for true targets with shape (batch_size,).
    - non_true_probs (torch.Tensor): Tensor of predicted probabilities for non-true targets with shape (batch_size, num_classes - 1).
    """
    # Use torch.arange to create indices for slicing
    indices = torch.arange(len(true_targets))

    # Index into the predicted probabilities tensor to get probabilities for true targets
    true_probs = probs[indices, true_targets]

    # Create a mask for the true targets
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[indices, true_targets] = 1

    # Index into the predicted probabilities tensor to get probabilities for non-true targets
    non_true_probs = probs[~mask].view(probs.size(0), probs.size(1) - 1)

    return true_probs, non_true_probs


# https://github.com/melanibe/failure_detection_benchmark/blob/c985d1a15da01f6e0c288d03e08a4d9b00895b85/failure_detection/evaluator.py#L27
def get_FPR_at_TPR(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95):
    fpr, fnr, thresholds = det_curve(y_true, scores)
    tpr = 1 - fnr
    index = np.argmin(tpr[tpr >= target_tpr])
    return fpr[index], tpr[index], thresholds[index]

def get_k(i, j):
    """
    This function maps a pair of indices (i, j) to a unique integer k.

    Args:
      i: The first index (0-based).
      j: The second index (0-based).

    Returns:
      An integer k representing the unique index for the given (i, j) pair.
    """
    if i == j:
        return -1
    if i>j:
        return get_k(j,i)
    elif i == 0 and j==1:
        return 0
    elif j == i+1:
        return get_k(i-1,9)+1
    else:
        return get_k(i, j-1)+1

def calc_outdist(W,logits, w21cache=None, num_classes=10):
    # print('calc_outdist')
    top2 = torch.topk(logits, 2)[1]
    t1 = top2[:,0]
    f1 = logits[torch.arange(len(logits)), t1]
    if w21cache is None:
        t2 = top2[:,1]
        w21star = torch.linalg.norm((W[t1]-W[t2]).detach(), ord=1, dim=1)
        f2 = logits[torch.arange(len(logits)), t2]
        dist = torch.abs(f1-f2)/w21star
    else:
        dmin = torch.ones_like(t1)*torch.inf
        for j in tqdm(range(num_classes)):
            t2 =torch.tensor([j]*len(t1))
            f2 = logits[torch.arange(len(logits)), t2]
            k = torch.tensor([get_k(i,j) for (i,j) in list(zip(t1,t2))])
            f1f2 = torch.where(k==-1, torch.inf, f1-f2)
            dist = torch.abs(f1f2)/w21cache[k]
            dmin = torch.where(dist<dmin, dist, dmin)
        return dmin