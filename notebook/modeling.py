import random
import numpy as np
from tqdm.notebook import tqdm as tqdm
import torch


def toDataset(X,y):
    return torch.utils.data.dataset.TensorDataset(X, y)

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

   
def create_sequential_with_depth(in_features, n_neurons, depth):
    layers = []
    layers.append(torch.nn.Linear(in_features, n_neurons))
    for _ in range(depth-2):
        layers.append(torch.nn.Linear(n_neurons, n_neurons))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(n_neurons, 1))
    
    return torch.nn.Sequential(*layers)

class CNET(torch.nn.Module):
    def __init__(self, in_features=10,
                 n_neurons=512, depth=4)->None:
        super().__init__()
        self.n_classes=2
        self.arch = create_sequential_with_depth(in_features, n_neurons, depth)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        out = self.arch(x)
        return torch.sigmoid(out)

class DeepNet:
    def __init__(self, model: torch.nn.Module,
                 device: str='cpu',
                 loss_fn: any=None,
                 optimizer: any=None,
                 num_epochs: int=10,
                 batch_size: int=256,
                 verbose: bool=True,
                 classifier: bool=False) -> None:
        super().__init__()
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device=device
        self.num_epochs = num_epochs
        self.verbose=verbose
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.batch_size = batch_size
        self.classifier=classifier

    def train(self, train_set, loader=False, seed=0):
        set_seeds(seed)
        if loader:
            train_loader=train_set
        else:
            train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)
        for epoch in tqdm(range(1, self.num_epochs+1)):
            avg_acc, avg_loss = self.train_epoch(train_loader)
            if self.verbose:
                print(f"epoch[{epoch}/{self.num_epochs}]: acc = {avg_acc:.2f} loss = {avg_loss:.5f}")

    def xtrain_epoch(self, train_loader)->None:
        self.model.train()
        loss_sum = 0.
        total = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # print(x.shape)
            logits = self.model(x)
            if len(logits.shape)==2 and self.model.n_classes<3:
                y = y.view(-1,1)
            pred = torch.sigmoid(logits-logits.flip(0))
            ytrue = (y > y.flip(0)).float()
            # print(pred.shape, ytrue.shape, pred, ytrue)
            loss =  self.loss_fn(pred, ytrue)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()*len(x)
            # ypred = logits.argmax(1)
            # correct = (y == ypred).sum().item()
            total += len(x)

        avg_loss = loss_sum/total
        avg_acc = 0. #correct/total
        return avg_acc, avg_loss
        
    def train_epoch(self, train_loader)->None:
        self.model.train()
        loss_sum = 0.
        # correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            # print(x.shape)
            logits = self.model(x)
            if len(logits.shape)==2 and self.model.n_classes<3:
                y = y.view(-1,1)
            loss =  self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()*len(x)
            total += len(x)

        avg_loss = loss_sum/total
        avg_acc = 0. #correct/total
        return avg_acc, avg_loss
    
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device))
    
    def test(self, test_set)->None:
        test_loader = torch.utils.data.DataLoader(test_set,
                                                    batch_size=self.batch_size,
                                                    shuffle=False)
        self.model.eval()
        correct = 0
        loss_sum=0.
        total=0
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                if len(logits.shape)==2 and self.model.n_classes<3:
                    y = y.view(-1,1)
                if self.classifier:
                    if self.model.n_classes>=3:
                        ypred = logits.argmax(1)
                    else:
                        ypred = logits>=0.5
                    correct = (y == ypred).sum().item()
                loss_sum = self.loss_fn(logits, y)*len(x)
                total += len(x)
            avg_test_loss = loss_sum/total
            test_acc = correct/total if self.classifier else 0
        return test_acc, avg_test_loss
