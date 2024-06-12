import os
import random

from matplotlib import pyplot as plt
import scipy
from tqdm.notebook import tqdm
from resnet import ResNet18
from scoreslib import *
from sklearn.metrics import auc as calc_auc, det_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score as calc_auroc
from robustbench.model_zoo.architectures.utils_architectures import ImageNormalizer
from robustbench.utils import load_model
# import sklearn
from torchvision import transforms

etf=torch.zeros(10, 10)
for c in range(10):
    for j in range(10):
        etf[c,j] = 1.0 if c==j else -1/9

scnames = [#'entropy',
#            'quadratic',
#            'spherical',
           'prob. margin', 
           'logit margin',
           'max prob.',
        #    'max logit',
        #    'energy',
        #    'doctor1',
        #    'doctor2',
           'dnet',
           'feature dist',
           'tcp',
           'tmargin',
           'output dist',
           'input dist']

def get_filenames_in_folder(folder_path):
    fd = os.listdir(folder_path)
    cond = lambda f: '.' not in f and not f.startswith('x')
        # and 'ResNet50' not in f# and 'loss' in f
    return [f for f in fd if cond(f)]

# print(models)
def get_code(model, dataset='cifar10', normthreat='Linf'):
    code=None
    if '2021' in model:
        if 'PA' in model:
            code= model[:2]
        elif 'Re' in model and '28_10_cutmix' in model:
            code= model[:2]+"8"
        else:
            code= model[:2]+"1"
    elif '2022' in model:
        if 'XC' in model:
            code= model[0]+model[-3]
        else:   
            code=model[:2]+"2"
    elif '2023' in model:
        code= model[:2]+model[-4]
    elif 'madry' in model or 'mart' in model:
        code= model[0]+model[2]
    else:
        code = model[:2]
    code = code+"0" if dataset == 'cifar10' else code+"1"
    if normthreat=='L2':
        code = code+"2"
    return code.upper()
    
def getps(model, code=False, dataset='cifar10', normthreat='Linf'):
    if code:
        return get_code(model, dataset=dataset, normthreat=normthreat)
    if 'cifar10' in model:
        ps = model.split('_')[1]
    else:
        ps = model.split('_')[0][:13]
    return ps

def lq2(y, t1=5, t2=95):
    y = np.array(y)
    q1=np.percentile(y, t1)
    q2=np.percentile(y, t2)
    idxs = (y > q1) & (y < q2)
    return idxs, y[idxs] 

def get_path(model, dataset='cifar10', normthreat='Linf'):
    prefix='_LinfC100' if dataset == 'cifar100' else '_Linf'
    return f"../xps/{dataset}/{model}{prefix}"



def fget_scores(model, dico, y=None, extra=False, dataset='cifar10', normthreat='Linf', w21cache=None):
    outpath=fget_out_path(model, extra=extra, dataset=dataset, normthreat=normthreat)
    outadvpath = fget_out_path(model, adv=True, extra=extra, dataset=dataset, normthreat=normthreat)
    out = torch.load(outpath, map_location=torch.device('cpu'))
    out_adv = torch.load(outadvpath, map_location=torch.device('cpu'))
    #
    featpath=get_feat_path(model, extra=extra, dataset=dataset, normthreat=normthreat)
    featadvpath = get_feat_path(model, adv=True, extra=extra, dataset=dataset, normthreat=normthreat)
    feat = torch.load(featpath, map_location=torch.device('cpu'))
    feat_adv = torch.load(featadvpath, map_location=torch.device('cpu'))
    # entropies = calc_entropy(out)
    # qscores = calc_qscore(out)
    pmargins = calc_pmargin(out)
    lmargins = calc_lmargin(out)
    # spherescores = calc_sphere(out)
    maxprobs = calc_maxprob(out)
    maxlogit = calc_maxlogits(out)
    # energy = calc_energy(out)
    # doctor1 = doctor_score1(out)
    # doctor2 = doctor_score2(out)
    # kl = calc_kldiv(out, out_adv)
    logitdist = calc_drep(out, out_adv)
    # tcp = torch.zeros_like(logitdist)
    # tmargins = torch.zeros_like(logitdist)
    dnet = torch.zeros_like(logitdist).uniform_()
    featuredist = calc_drep(feat, feat_adv)
    output_dist = get_outdist(model, out, dataset, normthreat=normthreat, w21cache=w21cache)
    if y is not None:
        tmargins = calc_tpmargin(out, y, from_logits=True)
        tcp = calc_tcp(out,y, from_logits=True)
    # scores_list = [entropies, qscores, spherescores,
    #                pmargins, lmargins, maxprobs,
    #                energy, doctor1, doctor2, logit_dist_score,
    #                dnet, logitdist, featuredist, tcp, tmargins]
    score_list = dict()
    # score_list['entropy'] = entropies
    # score_list['quadratic'] = qscores
    # score_list['spherical'] = spherescores
    score_list['prob. margin'] = pmargins
    score_list['logit margin'] = lmargins
    score_list['max prob.'] = maxprobs
    score_list['max logit'] = maxlogit
    # score_list['energy'] = energy
    # score_list['doctor1'] = doctor1
    # score_list['doctor2'] = doctor2
    score_list['dnet']  = dnet
    score_list['logit dist'] = logitdist
    score_list['feature dist'] = featuredist
    score_list['tcp'] = tcp
    score_list['tmargin'] = tmargins
    score_list['output dist'] = output_dist
    dico[model] = score_list

def get_targets_path(model, extra=False, dataset='cifar10', normthreat='Linf'):
    sufx = 'extra' if extra else ''
    sp = get_path(model, dataset=dataset, normthreat=normthreat)
    return f'{sp}/{sufx}fab_ytrueevm512.pt'

def fget_out_path(model, adv=False, extra=False, dataset='cifar10', normthreat='Linf'):
    sufx = 'extra' if extra else ''
    prefx = '_adv' if adv else ''
    sp = get_path(model, dataset, normthreat)
    return f'{sp}/{sufx}logits{prefx}_evm512.pt'

def get_feat_path(model, adv=False, extra=False, dataset='cifar10', normthreat='Linf'):
    sufx = 'extra' if extra else ''
    prefx = '_adv' if adv else ''
    sp = get_path(model, dataset, normthreat)
    return f'{sp}/{sufx}features{prefx}_evm512.pt'


####
def bin_separate(dfs, seuil, norm='norm inf'):
    y_binary = dict()
    for model, df in dfs.items():
        condition1 = df[norm] > seuil
        condition2 = df[norm] <= seuil        
        nonrobust = ~(condition1)
        np.testing.assert_array_equal(nonrobust, (condition2))
        y_binary[model] = nonrobust.to_numpy()
    return y_binary

def xbin_separate(dist, seuil):
    condition1 = dist > seuil
    nonrobust = ~(condition1)
    return nonrobust.to_numpy()

def get_AUCs(xmodels: list = None,
            y_binary: dict=None,
             balanced: bool = False,
            idxs = None,
            xscores=None,
            kdlcors=None):
    aurocscores = dict()
    auprscores = dict()
    fpr_tpr_values = dict()
    prec_rec_values = dict()
    fprattpr_values = dict()
    ratios = dict()
    for model in xmodels:
        auc_score = np.nan
        binvalues = y_binary[model]
        y_onehot_full = np.array(binvalues).astype(int) 
        
        negatives = (y_onehot_full==0).sum()
        positives = len(y_onehot_full)-negatives
        ratios[model]= negatives/(positives+negatives)
        assert negatives!=0 and positives!=0

        aurocs = dict()
        auprs=dict()
        fpr_tprs=dict()
        prec_recs=dict()
        fprattprs=dict()
        if idxs is None:
            idxs= np.arange(len(y_onehot_full))

        for k, scorename in enumerate(scnames):
            y_score = xscores[model][scorename]
            y_score = y_score[idxs]
            y_onehot = y_onehot_full[idxs]

            if kdlcors[model][scorename]>0: 
                y_score = -y_score
            fpr, tpr, tresholds = roc_curve(y_onehot, y_score)
            auroc = calc_auc(fpr, tpr)
            assert auroc == calc_auroc(y_onehot, y_score)
            prec, rec, _ = precision_recall_curve(y_onehot, y_score)
            aupr = calc_auc(rec, prec)
            fpr95, _,tr95 = get_FPR_at_TPR(y_onehot, y_score, 0.95)
            # fpr80, _,tr80 = get_FPR_at_TPR(y_onehot, y_score, 0.80)
            aurocs[scorename] = auroc
            auprs[scorename] = aupr
            fpr_tprs[scorename] = (fpr, tpr)
            prec_recs[scorename] = (prec, rec)
            fprattprs[scorename] = (fpr95, tr95)
        aurocscores[model] = aurocs
        auprscores[model]=auprs
        fpr_tpr_values[model] = fpr_tprs
        prec_rec_values[model] = prec_recs
        fprattpr_values[model] = fprattprs
    res = {'ratios': ratios, 
           'auroc': aurocscores,
           'aupr': auprscores,
           'fpr_tpr': fpr_tpr_values, 
           'prec_rec': prec_rec_values,
           'fprattpr': fprattpr_values
    }
        
    return res

def xget_AUCs(binvalues, scores, idxs=None):
    y_onehot_full = np.array(binvalues).astype(int) 
    negatives = (y_onehot_full==0).sum()
    positives = len(y_onehot_full)-negatives
    ratio = negatives/(positives+negatives)
    assert negatives!=0 and positives!=0

    if idxs is None:
        idxs= np.arange(len(y_onehot_full))

    y_score = scores[idxs]
    y_onehot = y_onehot_full[idxs]

    fpr, tpr, tresholds = roc_curve(y_onehot, y_score)
    auroc = calc_auc(fpr, tpr)
    assert auroc == calc_auroc(y_onehot, y_score)
    prec, rec, _ = precision_recall_curve(y_onehot, y_score)
    aupr = calc_auc(rec, prec)
    fpr95, _,tr95 = get_FPR_at_TPR(y_onehot, y_score, 0.95)
    # fpr80, _,tr80 = get_FPR_at_TPR(y_onehot, y_score, 0.80)
    res = {'ratio': ratio, 
           'auroc': auroc,
           'aupr': aupr,
           'fpr_tpr': (fpr, tpr), 
           'prec_rec': (prec, rec),
           'fprattpr': (fpr95, tr95)
    }
    return res


def regeval(regr, X_test, y_onehot):
    y_score = -regr.predict(X_test)
#     ypreds = np.array(y_test<0.5).astype(int)
    pr, rc, _ = precision_recall_curve(y_onehot, y_score)
    aupr_err = calc_auc(rc, pr)
    
    y_onehot_inv = (1 - y_onehot).astype(int)
    pr, rc, _ = precision_recall_curve(y_onehot_inv, -y_score)
    aupr_suc = calc_auc(rc, pr)
    
    auroc = calc_auroc(y_onehot, y_score)
    fpr95,_,_ = get_FPR_at_TPR(y_onehot, y_score)
    print(f"{sum(y_onehot)}/{len(y_onehot) - sum(y_onehot)} auroc:{auroc:.2f}, \
    aupr-error:{aupr_err:.2f}  aupr-success:{aupr_suc:.2f} fpr95:{fpr95:.2f}")


def plx(X, Y, M=10, label='-', shownumbers=False, color=None, bin_edges=None, fmt='o', ls='-'):
    if bin_edges is None:
        bin_edges = np.linspace(min(X), max(X), M+1)
    X_bins = np.digitize(X, bin_edges)
    # Prepare data for boxplot (group Y by bins)
    data_to_plot = [Y[X_bins == b] for b in range(1, M + 1)]
    xdata_to_plot = [X[X_bins == b] for b in range(1, M + 1)]
    means = [np.median(data) for data in data_to_plot if len(data)>0]
    xmeans = [np.median(data) for data in xdata_to_plot if len(data)>0]
    std_errors = [np.array(data).std(ddof=1)/np.sqrt(len(data)) for data in data_to_plot if len(data)>0]
    plt.errorbar(xmeans, means, std_errors,
        fmt=fmt, capsize=3,markersize=8,
        linestyle=ls, label=label, color=color)
#     plt.scatter(X, Y)
    if shownumbers:
        sizes = [len(data) for data in data_to_plot]
        for i in range(len(means)):
            plt.annotate(f"{sizes[i]}", (xmeans[i], means[i]))

def plot_bar(model, scorename, scoreref, labelpos, labelneg,tr,
             ufs, ufs_ref, positiveflags):
########
    positiveflags = np.array(positiveflags)
    scoresign = np.sign(sum(ufs))
    plt.figure()
    plt.axvline(tr, color='b', linestyle='dashed', linewidth=2)
    X1, Y1 = ufs_ref[positiveflags], scoresign*ufs[positiveflags]
    X2, Y2 = ufs_ref[~positiveflags], scoresign*ufs[~positiveflags]
    plx(X1, Y1, label=labelpos, shownumbers=False)
    plx(X2, Y2, label=labelneg, shownumbers=False)
    xlabel = "True Class Margin" if scoreref == 'tmargin' else scoreref
    plt.xlabel(xlabel)
    plt.ylabel(scorename)
    title=f"{getps(model)}"
    plt.title(title)
    plt.legend(loc='lower right')
    savetag=f"{scorename}_bar_tmargin.pdf"
    plt.savefig(savetag)
    plt.show()
    plt.close()

class ResNetNormed(torch.nn.Module):
    def __init__(self, model, mean=None, std=None):
        super().__init__()
        if mean is None or std is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
        self.transfrom = transforms.Normalize(mean = mean, std = std)
        self.model=model
        # self.normalization_layer = NormalizationLayer(mean, std)

    def forward(self, x):
        out = self.transfrom(x)
        logits = self.model(out)
        return logits
    
def get_model(model_name, dataset='cifar10', thread_norm='Linf'):
    model = load_model(model_name=model_name,
                        dataset=dataset,
                        threat_model=thread_norm)
    return model

def fc_head_weight(model):
    if isinstance(model, ResNetNormed):
        name, module = list(model.named_children())[-1]
        m = getattr(model, name)
        name, fc_head = list(module.named_children())[-1]
    else:
        name, fc_head = list(model.named_children())[-1]
    return fc_head.weight

def get_fc_head_weight(model_name,dataset='cifar10', thread_norm='Linf'):
    model = get_model(model_name, dataset, thread_norm)
    if model_name in ['XCiT']:# or isinstance(model[0], ImageNormalizer):
        return fc_head_weight(model[1])
    else:
        return fc_head_weight(model)
    
def get_outdist(model, out, dataset='cifar10', normthreat='Linf', w21cache=None):
    W = get_fc_head_weight(model, dataset, thread_norm=normthreat)
    num_classes = int(dataset[5:]) if dataset!='imagenet' else 1000
    dist= calc_outdist(W, out, w21cache=w21cache, num_classes=num_classes)
    return dist


def get_rob_acc(logit_margins, lda_treshold, correct_pred=None):
    vulnerable = -logit_margins>=lda_treshold
    if correct_pred is not None:
        RobAcc = 100*(correct_pred & ~vulnerable).float().mean()
    else:
        RobAcc = 100*(~vulnerable).float().mean()
    return RobAcc

def determine_lda(logit_margins_subset, y_true_subset, correct_pred_subset, alpha_trh=None):
    if correct_pred_subset is None:
        rob_subset = 100*(~torch.tensor(y_true_subset)).float().mean()
    else:
        rob_subset = 100*(correct_pred_subset & ~y_true_subset).float().mean()
    
    if alpha_trh is not None:
        ## the threshold for alpha_trh TPR
        lda_treshold = get_FPR_at_TPR(y_true_subset, -logit_margins_subset, alpha_trh)[-1]
    else:
        lda_treshold, best_approx= np.inf, np.inf
        for alpha in [0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.99]:
            lda_trh = get_FPR_at_TPR(y_true_subset, -logit_margins_subset, alpha)[-1]
            approx = abs(rob_subset - get_rob_acc(logit_margins_subset, lda_trh, correct_pred_subset))
            if approx < best_approx:
                lda_treshold = lda_trh
                best_approx = approx
    return lda_treshold

def SampleEfficientRobustAccuracy(margins, logits,
                                  labels=None, epsilon=8/255,
                                  n_subset=500, alpha_trh=None):
    correct_pred = logits.max(1)[1].eq(labels) if labels is not None else None
    logit_margins = calc_lmargin(logits)
    #sample n_subset samples uniformly at random
    idxs = np.random.choice(np.arange(len(margins)), size=n_subset, replace=False)
    cor_on_subset = scipy.stats.kendalltau(margins[idxs], logit_margins[idxs])[0]
    #create vulnerability ground thruths at epsilon robustness threshold
    y_true_subset = margins[idxs] <= epsilon
    correct_pred_subset = correct_pred[idxs] if labels is not None else None
    #determine the threshold lda for epsilon
    lda_treshold = determine_lda(logit_margins[idxs], y_true_subset, 
                                 correct_pred_subset, alpha_trh=alpha_trh)
    #determine the non-vulnerable samples on whole data based on lda_treshold
    RobAcc = get_rob_acc(logit_margins, lda_treshold, correct_pred)
    return float(RobAcc), cor_on_subset