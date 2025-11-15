import os
import errno
import torch
import torch.nn as nn
import torch.nn.init as init
import json

__all__ = ['mkdir_p', 'AverageMeter', 'init_params', 'save_metrics_json']

def mkdir_p(path):
    """Creates a directory if it does not exist."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter:
    """
    Computes and stores the average and current values.
    Useful for tracking loss, accuracy, and calibration metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all stored values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def init_params(model):
    """
    Initializes model parameters with He initialization for Conv layers
    and Normal initialization for Linear layers.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.zeros_(m.bias)

def save_metrics_json(metrics_dict, filepath):
    '''Saves a dictionary of metrics to a JSON file'''
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {filepath}")
    except Exception as e:
        print(f"Error saving metrics to {filepath}: {e}")    