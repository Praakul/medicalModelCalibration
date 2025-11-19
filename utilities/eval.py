import torch
import numpy as np
from typing import List, Tuple, Dict


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after given patience."""
    def __init__(self, patience: int = 10, verbose: bool = True, delta: float = 1e-4):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def step(self, val_loss: float, model: torch.nn.Module = None, path: str = None) -> bool:
        """
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module, optional): Model to save
            path (str, optional): Path to save model checkpoint
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
            if model is not None and path is not None:
                torch.save(model.state_dict(), path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss
            self.counter = 0
            if model is not None and path is not None:
                torch.save(model.state_dict(), path)

        return self.early_stop

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """
    Computes top-k accuracies for different values of k
    
    Args:
        output: Model predictions (N, C) where C is number of classes
        target: Ground truth labels (N,)
        topk: Tuple of k values to compute accuracy for
    
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        maxk = min(output.shape[-1], maxk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader,
                  device: str,
                  num_classes: int) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics
    
    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes in dataset
    
    Returns:
        Dictionary containing various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    acc = accuracy(all_preds, all_targets)[0].item()
    probs = torch.softmax(all_preds, dim=1)
    preds = torch.argmax(probs, dim=1)
    f1 = calculate_f1_score(preds.numpy(), all_targets.numpy(), num_classes)  
    auc = calculate_auc(probs.numpy(), all_targets.numpy(), num_classes)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'auc': auc
    }
    return metrics        
        

def calculate_f1_score(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Calculate macro F1 score"""
    from sklearn.metrics import f1_score
    return f1_score(targets, preds, average='macro')

def calculate_auc(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """Calculate AUC score (one-vs-rest for multiclass)"""
    from sklearn.metrics import roc_auc_score
    if num_classes == 2:
        return roc_auc_score(targets, probs[:, 1])
    else:
        return roc_auc_score(targets, probs, multi_class='ovr')

