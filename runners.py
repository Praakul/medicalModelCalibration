import torch
from tqdm import tqdm
from utilities.metrics import CalibrationMetrics  
from utilities.eval import accuracy
from utilities.misc import AverageMeter

def train(trainloader, model, optimizer, criterion):
    """ Train the model for one epoch """
    model.train().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    bar = tqdm(enumerate(trainloader), total=len(trainloader), desc='Training')
    for batch_idx, (inputs, targets) in bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        prec1, = accuracy(outputs, targets, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        
        bar.set_postfix(loss=losses.avg, top1=top1.avg)
    
    return losses.avg, top1.avg

def test(testloader, model, criterion):
    """ Evaluate the model on the test set """
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = []
    all_outputs = []

    model.eval().cuda()
    
    bar = tqdm(enumerate(testloader), total=len(testloader), desc='Testing')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            prec1, prec3, prec5 = accuracy(outputs, targets, topk=(1, 3, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top3.update(prec3.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            
            all_targets.append(targets)
            all_outputs.append(outputs)
    
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    
    return all_outputs, all_targets
