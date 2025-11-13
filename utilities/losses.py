import torch
import torch.nn as nn
import logging
from argparsor import parse_args as args

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma= 2, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):
        target = target.view(-1,1)
        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        loss = torch.tensor(0.0,device=output.device)
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, loss ='focal', beta = 5 , gamma = 2):
        super(CombinedLoss, self).__init__()
        self.beta = beta
        
        if loss == 'cross_entropy':
            
            self.primary_loss = CrossEntropy()
        elif loss == 'focal':
            self.primary_loss = FocalLoss()
        else:
            raise ValueError(f"Unsupported Primary loss: {loss}")
        print(loss)
        self.mdca_loss = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.primary_loss(logits, targets)
        loss_cal = self.mdca_loss(logits, targets)
        return loss_cls + self.beta * loss_cal

# Loss dictionary with only the required losses
loss_dict = {
    "cross_entropy": CrossEntropy,
    "focal_loss": FocalLoss,
    "NLL+MDCA": lambda gamma, beta: CombinedLoss(loss="cross_entropy", beta= beta, gamma = gamma),  
    "FL+MDCA": lambda gamma, beta: CombinedLoss(loss="focal", beta=beta, gamma=gamma) 
}


