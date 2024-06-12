import os
import argparse
import sys
from time import time
import random
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
import torch
from robustbench.utils import load_model
from autoattack import AutoAttack


from art.estimators.classification.pytorch import PyTorchClassifier
from art.metrics.metrics import clever_u

import plotly.express as px

from regularization.distributed_train import load_checkpoint
from regularization.pkg.dataset import get_CIFAR10, get_CIFAR100, get_ImageNetVal
from regularization.pkg.mutils import ResNetNormed
from regularization.pkg.resnet import ResNet18



def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the models for this experiment."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Test mode"
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="Test mode"
    )
    parser.add_argument(
        "--dataset-path",
        default="/home-local2/jongn2.extra.nobkp/data",
        help="the path to the dataset",
        type=str,
    )
    parser.add_argument(
        "--cpus-per-trial",
        default=1,
        help="the number of CPU cores to use per trial",
        type=int,
    )
    parser.add_argument(
        "--project-name",
        help="the name of the Weights and Biases project to save the results",
        default='evalrobustness',
        # required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="dataset used",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--model-name",
        help="model to use",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--paramfile",
        help="param file to use",
        default="",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--debug-strategy",
        help="the strategy to use in debug mode",
        default="Random",
        type=str,
    )
    return parser.parse_args(args)

class ModelWrapper(torch.nn.Module):
    fc_head: torch.nn.Linear
    def __init__(self, model):
        super().__init__()
        if isinstance(model, ResNetNormed):
            name, module = list(model.named_children())[-1]
            m = getattr(model, name)
            name, fc_head = list(module.named_children())[-1]
            setattr(m, name, torch.nn.Identity())
        else:
            name, fc_head = list(model.named_children())[-1]
            setattr(model, name, torch.nn.Identity())
        self.embedding = model 
        self.e1 = None
        self.fc_head = fc_head

    def forward(self, x):
        self.e1 = self.embedding(x)
        return self.fc_head(self.e1)

    def get_embedding(self):
        if self.e1 is not None:
            e1 = self.e1.squeeze().clone()
            self.e1 = None
            return e1
        else:
            raise ValueError('Forward should be executed first')

    def get_embedding_dim(self):
        return self.fc_head.in_features


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_model(model_name, params):
    if 'loss' in model_name.lower():
        model = get_rg_model(model_name, './regularization/checkpoints')
    else:
       model = get_prt_model(model_name, params)
    return model

def get_rg_model(model_name, path='./regularization/checkpoints'):
    model = ResNet18()
    ckptpath = f'{path}/{model_name}.pt'
    model,_ = load_checkpoint(model,ckptpath, intest=True)
    return model

def get_prt_model(model_name, params):
    model = load_model(model_name=model_name,
                        dataset=params['dataset_name'],
                        threat_model=params['norm_thread'])
    return model

def check_acc(model, test_loader, device):
  total = 0
  correct = 0
  for x, y in tqdm(test_loader):
    x, y = x.to(device), y.to(device)
    pred = torch.max(model(x), 1)[1]
    correct += (y==pred).sum().item()
    total += len(y)
  print(f'XAccuracy : {100*correct/total:.2f}')

def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """
      #
    attack = params['attack']
    batch_id=params['batch_id']
    norm_thread = params['norm_thread']
    model_name = params['model_name'] if args.model_name is None else args.model_name
    if args.dataset is not None:
        params['dataset_name'] = args.dataset
    root= params['results_root_path']
    new = "_New" if params['unsup_version'] else ""
    c100 = "C100" if params['dataset_name']=='cifar100' else ''
    resultsDirName = f"{root}/{params['dataset_name']}/{model_name}_{norm_thread}{new}{c100}"
    source_path= f"{root}/{params['dataset_name']}/{model_name}_{norm_thread}{new}{c100}"

    if not os.path.exists(resultsDirName):
        os.makedirs(resultsDirName)
        print("Results directory ", resultsDirName,  " Created ")
 
    set_seeds(params['seed'])
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda" if use_cuda else "cpu")
    print(f'Using GPU : {use_cuda}')

    """ MODEL """
    #Load Model
    print(model_name)
    model = get_model(model_name, params)
    model = ModelWrapper(model)
    model = model.to(device)
    model.eval()
    print("Model Loaded")
    ##### Dataset #####
    if params['dataset_name'] == 'cifar10':
        test_set = get_CIFAR10(train=False)
        train_set = get_CIFAR10(train=True)
        valid_size= 5000
        _ , val_set = torch.utils.data.random_split(train_set, [len(train_set) - valid_size, valid_size])
    elif params['dataset_name'] == 'cifar100':
        test_set = get_CIFAR100(train=False)
        train_set = get_CIFAR100(train=True)
        valid_size= 5000
        _ , val_set = torch.utils.data.random_split(train_set, [len(train_set) - valid_size, valid_size])
    elif params['dataset_name'] == 'imagenet':
        datapath='/home-local2/jongn2.extra.nobkp/imagenet'
        test_set = get_ImageNetVal(datapath)
        if params['task'] != 'get_test_logits':
            subsetsize= 1000
            _ , test_set = torch.utils.data.random_split(test_set, 
                                     [len(test_set) - subsetsize, subsetsize])
        val_set = None


    save_tag = 'evm512' #
    batch_size = params['batch_size'] if  attack != 'clever' else 1
    if params['is_train']:
        train_tag = 'extra'
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    else:
        train_tag = ''
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2) 
    ##
    ##xp test
    if params['task'] in ['get_valid_logits', 'get_test_logits']:
        t_set = val_set if 'valid' in params['task'] else test_set
        print(len(t_set))
        acc, val_out = get_logits(model, t_set, device, params['batch_size'])
        print(f"{params['task']}/Acc.  {100*acc:.2f}")
        torch.save(val_out, os.path.join(resultsDirName, f'val_logit_{save_tag}.pt'))
        torch.save(t_set.targets, os.path.join(resultsDirName, f'val_ytrue_{save_tag}.pt'))
        exit(0)
    #
    check_acc(model, test_loader, device)

    if attack == 'clever':
        save_tag = f"evm512_{batch_id}"
        t0 = time()
        clscores = get_clever_scores(model, test_loader,  norm_thread, batch_id=batch_id)
        t1 = time()
        torch.save(torch.tensor(clscores), os.path.join(resultsDirName,f'{train_tag}clever_score_{save_tag}.pt'))
        print(len(clscores), t1-t0, clscores.mean())
        write_time(t1-t0, resultsDirName, save_tag)
        exit(0)
    ##
    print("..getXY..")
    original, targets, pred_targets = get_XY(model, test_loader,device) 
    print(len(original))
    ##TIMING
    if params['task']=='timming':
        timming_eval(model, original[:128], pred_targets[:128], norm_thread, device, model_name)
        exit(0)
    ##
    path = f"{source_path}/{train_tag}{attack}_adverserial{save_tag}.pt"
    if os.path.exists(path):
        adversarial, y_adversarial = restore_adv(source_path, params,train_tag, save_tag, device)
    else:
        if attack == 'fab':
            attack_fn = ifab_attack if params['dataset_name']=='imagenet' else fab_attack
        else:
            attack_fn = auto_attack
        print(f"{attack} attack...")
        
        # exit(0)
        if params['dataset_name']=='imagenet':
            adversarial, y_adversarial = attack_fn(model,test_loader, norm_thread, device, resultsDirName, save_tag)
        else:
            adversarial, y_adversarial = attack_fn(model,
                                                original,
                                                pred_targets if params['unsup_version'] else targets,
                                                norm_thread,
                                                device)
           
        ##
        torch.save(adversarial, os.path.join(resultsDirName,f'{train_tag}{attack}_adverserial{save_tag}.pt'))
        torch.save(y_adversarial, os.path.join(resultsDirName,f'{train_tag}{attack}_y_adverserial{save_tag}.pt'))
        try:
            acc, ypred = assert_yadvcoherent(adversarial, y_adversarial, model, device)
            torch.save(ypred, os.path.join(resultsDirName,f'{train_tag}{attack}_y_adverserial{save_tag}.pt'))
            write_adv_acc(acc, resultsDirName, save_tag+f"_{attack}testing")
        except:
            print("assert y_advcorrent failed.") 
    torch.save(targets, os.path.join(resultsDirName,f'{train_tag}{attack}_ytrue{save_tag}.pt'))
    print(original.shape, adversarial.shape, targets.shape)
    idxs = torch.arange(len(original))[:len(adversarial)]
    my_dataset = torch.utils.data.dataset.TensorDataset(original.cpu()[idxs],
                                                        adversarial.cpu(),
                                                        targets.cpu()[idxs])
    eval_part(model, device, params, my_dataset, resultsDirName, train_tag, save_tag, norm_thread)

##
def timming_eval(model, original, targets, norm_thread, device, model_name):
    t0 = time()
    _, _, _ = fab_attack(model,
                        original,
                        targets,
                        norm_thread,
                        device)  
    t1 = time()
    elapsed_time = t1-t0
    print(f'Elapsed Time {model_name}: {elapsed_time:.2f}s')
    with open(os.path.join('./','timming.txt'), "a") as f:
        f.write(f"{model_name}: {elapsed_time:.3f}\n")
        f.close()

def write_time(elapsed_time, resultsDirName, save_tag):
    print(f'Elapsed Time {save_tag}: {elapsed_time:.2f}s')
    with open(os.path.join(resultsDirName,f'result{save_tag}.txt'), "a") as f:
        f.write(f"Elasped Time (batch {save_tag}): {elapsed_time:.2f}")
        f.close()

def get_XY(model, test_loader,device='cuda'):
    x_test, y_test, pred_y_test = [], [], []
    for x, y in tqdm(test_loader):
        pred_y = model(x.to(device)).max(1)[1]
        x_test.append(x)
        y_test.append(y)
        pred_y_test.append(pred_y)
    return torch.cat(x_test), torch.cat(y_test), torch.cat(pred_y_test)

def assert_yadvcoherent(x_adv, y_adv, model, device):
    print(x_adv.shape, y_adv.shape)
    data = torch.utils.data.dataset.TensorDataset(x_adv.cpu(), y_adv.cpu())
    dataloader = torch.utils.data.DataLoader(data, batch_size=128,
                                             shuffle=False, num_workers=1)
    ypredlist = []
    correct = 0.
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        ypred  = out.max(1)[1]
        ypredlist.append(ypred)
        correct += (ypred == y).sum()
        total += len(ypred)
    ypred = torch.cat(ypredlist)
    try:
        torch.testing.assert_close(ypred, y_adv)
        print("test success: y_adv == out_adv")
        return correct/total, ypred
    except:
        print("test fail")
        return correct/total, ypred

def restore_adv(source_path, params, train_tag, save_tag, device):
    advlist = []
    yadvlist = []
    adv =torch.load(f"{source_path}/{train_tag}{params['attack']}_adverserial{save_tag}.pt", map_location=device)
    advlist.append(adv)
    yadv =torch.load(f"{source_path}/{train_tag}{params['attack']}_y_adverserial{save_tag}.pt", map_location=device)
    yadvlist.append(yadv)
    adv=torch.cat(advlist)
    yadv = torch.cat(yadvlist)
    return adv, yadv

def write_adv_acc(adv_acc, resultsDirName, save_tag):
    print(f'adv acc {save_tag}: {100*adv_acc:.2f}%')
    with open(os.path.join(resultsDirName,f'result{save_tag}.txt'), "w") as f:
        f.write(f"Adverserial accuracy (batch {save_tag}): {adv_acc}")
        f.close()

def fab_attack(model, x_test, y_test, norm_thread, device):
    max_eps =np.inf 
    adversary = AutoAttack(model, 
                           norm=norm_thread,
                           eps=max_eps, version='custom',
                           attacks_to_run=['fab'],
                           device=device)
    x_adv, y_adv = adversary.run_standard_evaluation(x_test.to(device),
                                                     y_test.to(device), return_labels=True,
                                                     bs=128)
    return x_adv, y_adv

def ifab_attack(model, test_loader, norm_thread, device,resultsDirName, save_tag, **args):
    max_eps =np.inf
    adversary = AutoAttack(model, 
                           norm=norm_thread,
                           eps=max_eps, version='custom',
                           attacks_to_run=['fab'],
                           device=device)
    x_advs = []
    y_advs = []
    ytargets = []
    for x, y  in test_loader:
            x = x.to(device)
            pred_y = model(x).max(1)[1]
            x_adv, y_adv = adversary.run_standard_evaluation(x, pred_y, return_labels=True, bs=3)
            #
            x_advs.append(x_adv.detach().cpu())
            y_advs.append(y_adv.detach().cpu())
            ytargets.append(y)
            #
            x_adv = torch.cat(x_advs, dim=0)
            y_adv = torch.cat(y_advs)
            ytarget = torch.cat(ytargets)
            torch.save(x_adv, os.path.join(resultsDirName,f'fab_adverserial{save_tag}.pt'))
            torch.save(y_adv, os.path.join(resultsDirName,f'fab_y_adverserial{save_tag}.pt'))
            torch.save(ytarget, os.path.join(resultsDirName,f'fab_ytrue{save_tag}.pt'))
    x_adv = torch.cat(x_advs, dim=0)
    y_adv = torch.cat(y_advs)
    return x_adv, y_adv

def auto_attack(model, x_test, y_test, norm_thread, device, unsup_version=True):
    max_eps = 8/255
    adversary = AutoAttack(model, 
                           norm=norm_thread,
                           eps=max_eps, version='standard',
                           device=device)
    x_adv, y_adv = adversary.run_standard_evaluation(x_test.to(device),
                                                     y_test.to(device), return_labels=True)
    return x_adv, y_adv

def get_clever_scores(model, test_loader,  norm_thread, batch_id=None, minpixel=0., maxpixel=1.0):
    clever_args={'min_pixel_value':  minpixel,
                'max_pixel_value': maxpixel,
                'nb_batches':100,
                'batch_size':128,
                'norm':float(norm_thread[1:]), # eg: 'L2'[1:]=2, 'Linf'[1:]=inf
                'radius':.20,
                'pool_factor':5}
    cl_scores = []
    model.eval()
    n=500
    for i, data in tqdm(enumerate(test_loader)):
        print(i)
        if i < batch_id*n and i>(batch_id+1)*n:
            continue
        clever_dis = clever_score_u(model, data[0][0], **clever_args)
        cl_scores.append(clever_dis)
        if len(cl_scores)==n:
            break
    return np.array(cl_scores)

def clever_score_u(model, x, **args):
    classifier = PyTorchClassifier(
    model=model,
    clip_values=(args['min_pixel_value'], args['max_pixel_value']),
    loss=None,
    optimizer=None,
    input_shape=(1, 32, 32),
    nb_classes=10,
    )
    res = clever_u(classifier, x.numpy(), 
                    nb_batches=args['nb_batches'], 
                    batch_size=args['batch_size'], 
                    radius=args['radius'], 
                    norm=args['norm'], 
                    pool_factor=args['pool_factor'])
    return res


def eval_part(model: ModelWrapper, device, params, my_dataset, resultsDirName, train_tag, save_tag, norm_thread):
    acc, adv_acc, adv_failure, df, logits, logits_adv, features, features_adv  = eval(model, my_dataset, device, params)
    # assert model.get_embedding_dim() == features.shape[1]
    torch.save(logits, os.path.join(resultsDirName,f'{train_tag}logits_{save_tag}.pt'))
    torch.save(logits_adv, os.path.join(resultsDirName,f'{train_tag}logits_adv_{save_tag}.pt'))

    torch.save(features, os.path.join(resultsDirName,f'{train_tag}features_{save_tag}.pt'))
    torch.save(features_adv, os.path.join(resultsDirName,f'{train_tag}features_adv_{save_tag}.pt'))

    df["(Acc,Adv Acc, Adv)"]=df[["Acc", "Adv Acc","Adv"]].apply(tuple, axis=1)
    df.to_csv(os.path.join(resultsDirName,f"{train_tag}{params['attack']}_results_{save_tag}.csv"))

    y_fig = f"norm {norm_thread[1:]}"
    fig=px.scatter(df,x="Entropy", y=y_fig, color="(Acc,Adv Acc, Adv)", marginal_x="histogram", marginal_y="histogram")
    fig.write_html(os.path.join(resultsDirName,f"{train_tag}{params['attack']}_vs_entropy.html"))

    fig = px.density_contour(df, x="Entropy", y=y_fig, marginal_x="histogram", marginal_y="histogram")
    fig.write_html(os.path.join(resultsDirName,f"{train_tag}{params['attack']}_vs_entropy_ctdsity.html"))

    df_adv=df[df["Adv"]==True]
    print(f'correl on {len(df_adv)} elements')
    spearman=df_adv[["Entropy",f"norm {norm_thread[1:]}"]].corr("spearman")
    print(spearman)
    print(spearman.iloc[0,1])
    print(f'adv acc: {100*adv_acc:.2f}%')
    with open(os.path.join(resultsDirName,f"{train_tag}{params['attack']}_result_all.txt"), "w") as wf:
      wf.write(f"Accuracy: {acc} \n")
      wf.write(f"Adverserial accuracy: {adv_acc}\n")
      wf.write(f"Adverserial failure: {adv_failure}\n")
      wf.write(f"Spearman correlation Entropy vs {params['attack']} norm (adverserial only): {spearman.iloc[0,1]}\n")
      wf.close()

def get_logits(model, my_dataset, device, bs=128):
    correct=0
    total=0
    logits_list = []
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=bs,
                                              shuffle=False, num_workers=2)
    for images, target in tqdm(loader):
        images, target = images.to(device), target.to(device)
        with torch.no_grad():
          out = model(images)
          logits_list.append(out)
          _, ypred = torch.max(out.data, 1)
          correct += sum(ypred == target).item()
          total += len(ypred)

    acc = correct/total
    out = torch.cat(logits_list, dim=0)
    return acc, out

def get_feature_distance(model: ModelWrapper, feat: torch.Tensor, top2: torch.Tensor):
    #TODO: Not implemented here 
    return torch.zeros_like(feat)

def eval(model: ModelWrapper, my_dataset, device, params):
    correct=0
    correct_adv=0
    constant=0
    # hs=[]
    # norms=[]
    total=0
    df_list=[]
    features_list = []
    features_adv_list = []    
    logits_list = []
    logits_adv_list = []
    adv_flag = []
    loader = torch.utils.data.DataLoader(my_dataset, batch_size=params['batch_size'],
                                              shuffle=False, num_workers=2)
    for i, (images, adv_images, target) in enumerate(loader):
        images, adv_images, target = images.to(device), adv_images.to(device), target.to(device)
        with torch.no_grad():
          out = model(images)
          logits_list.append(out)
          feat=model.get_embedding()
          features_list.append(feat)
        #   assert model.get_embedding_dim() == feat.shape[1]

          out_adv = model(adv_images)
          logits_adv_list.append(out_adv)
          feat_adv=model.get_embedding()
          features_adv_list.append(feat_adv)
        #   assert model.get_embedding_dim() == feat.shape[1]
          
          entropy = torch.special.entr(torch.softmax(out,1)).sum(1)
          adv_entropy = torch.special.entr(torch.softmax(out_adv,1)).sum(1)
          norm2 = torch.linalg.vector_norm(images.flatten(1) - adv_images.flatten(1),ord=2,dim=1)
          norm_inf = torch.linalg.vector_norm(images.flatten(1) - adv_images.flatten(1),ord=np.inf,dim=1)
          #
          top2 = torch.topk(out, 2)
          norm_feat= get_feature_distance(model, feat, top2)

          _, ypred = torch.max(out.data, 1)
          _, y_adv = torch.max(out_adv.data, 1)

          correct_adv += sum(y_adv == target ).item()
          constant += sum(y_adv == ypred ).item()
          correct += sum(ypred == target).item()
          total += len(y_adv)
          acc =(ypred == target).cpu().numpy()
          adv_acc =(y_adv == target).cpu().numpy()
          adv = (ypred != y_adv).cpu().numpy()
        #   if sum(adv)!= 128:
        #     print(i, sum(adv), y[~adv].cpu(), y_adv[~adv].cpu(), target[~adv].cpu())
          adv_flag.append(adv)
          df_list+=zip(entropy.cpu().numpy(), adv_entropy.cpu().numpy(),
                        norm2.cpu().numpy(), norm_inf.cpu().numpy(), norm_feat.cpu().numpy(),
                        acc, adv_acc ,adv )
    df=pd.DataFrame(df_list, columns=["Entropy", "Adv Entropy", "norm 2",
                                      "norm inf","feat norm", "Acc", "Adv Acc", "Adv"])
   
    print(f'constant {constant}/10k')
    adv_flag = np.concatenate(adv_flag)
    print(f'constant {constant}/10k, adv {sum(adv_flag)}')

    logits = torch.cat(logits_list, dim=0)
    logits_adv = torch.cat(logits_adv_list, dim=0)

    features = torch.cat(features_list, dim=0)
    features_adv = torch.cat(features_adv_list, dim=0)

    y_adv = logits_adv.max(1)[1]
    ypred = logits.max(1)[1]
    y = my_dataset.tensors[2].to(device)
  
    acc, adv_acc, adv_failure = correct/total, correct_adv/total, constant/total
    advf = (ypred == y_adv).sum()/len(ypred)
    print("##", (ypred == y_adv).sum().item(), (ypred == y).sum().item(), (y == y_adv).sum().item())
    try:
       torch.testing.assert_close(torch.tensor(adv_failure), advf, check_device=False)
       print(f'test in eval success {adv_failure:.5f},{advf:.5f}')
    except AssertionError:
      print(torch.tensor(adv_failure), advf)
      print(f'test in eval failed {adv_failure:.4f},{advf:.4f},{torch.isclose(advf, torch.tensor(adv_failure))}')
    alladvtrue = df['Adv'].sum()#/len(df))
    print(f"all adv {alladvtrue:.5f}")
    if alladvtrue == 10000:
      print(f"df correct {alladvtrue:.5f}")
    else:
      print(f"df wrong {alladvtrue:.5f}")
    return acc, adv_acc, adv_failure, df, logits, logits_adv, features, features_adv

def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {}

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)

def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    paramsfilename = f'./params{args.paramfile}.yaml'
    with open(paramsfilename, 'r') as param_file:
        params = yaml.load(param_file, Loader=yaml.SafeLoader)
    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()