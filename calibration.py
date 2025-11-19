import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import argparse 
from pathlib import Path 
from utilities.misc import mkdir_p, save_metrics_json
from utilities.__init__ import save_checkpoint
from utilities.data_loader import get_data_loaders
from models.resnet import build_model
from utilities.metrics import CalibrationMetrics

class PostHocCalibrator(nn.Module):
    """
    Base class for post-hoc calibration models.
    It freezes the base_model by default.
    """
    def __init__(self, base_model):
        super(PostHocCalibrator, self).__init__()
        self.base_model = base_model
        self._freeze_model()

    def _freeze_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)

class TemperatureScaling(PostHocCalibrator):
    def __init__(self, base_model):
        super(TemperatureScaling, self).__init__(base_model)
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.T

    def calibrate(self, val_loader):
        """
        Optimizes the temperature T on the validation set using LBFGS.
        """
        criterion = nn.CrossEntropyLoss()
        
        logging.info("Caching logits from validation set...")
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.cuda(), targets.cuda()
                all_logits.append(self.base_model(images))
                all_targets.append(targets)
        all_logits = torch.cat(all_logits).cuda()
        all_targets = torch.cat(all_targets).cuda()

        optimizer = optim.LBFGS([self.T], lr=0.01, max_iter=100)

        def closure():
            optimizer.zero_grad()
            self.T.data.clamp_(min=1e-4) 
            scaled_logits = all_logits / self.T
            loss = criterion(scaled_logits, all_targets)
            loss.backward()
            return loss

        logging.info("Optimizing temperature T...")
        optimizer.step(closure)
        self.T.data.clamp_(min=1e-4) 
        
        logging.info(f"Optimal Temperature: {self.T.item()}")
        return self.T.item()

class DirichletScaling(PostHocCalibrator):
    def __init__(self, base_model, num_classes, Mu=0.0):
        super(DirichletScaling, self).__init__(base_model)
        self.num_classes = num_classes
        self.scaling_layer = nn.Linear(num_classes, num_classes)
        self.Mu = Mu 

    def forward(self, x):
        logits = self.base_model(x)
        scaled_logits = self.scaling_layer(logits)
        return scaled_logits

    def regularizer(self):
        W, b = self.scaling_layer.parameters()
        b_loss = ((b**2).sum()) / self.num_classes
        return self.Mu * b_loss

    def loss_func(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets) + self.regularizer()

    def calibrate(self, val_loader):
        """
        Fits the linear scaling layer on the validation set using LBFGS.
        This finds the single best set of parameters to minimize NLL
        without overfitting.
        """
        criterion = nn.CrossEntropyLoss()
        
        logging.info("Caching logits from validation set...")
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.cuda(), targets.cuda()
                all_logits.append(self.base_model(images))
                all_targets.append(targets)
        all_logits = torch.cat(all_logits).cuda()
        all_targets = torch.cat(all_targets).cuda()
        optimizer = optim.LBFGS(self.scaling_layer.parameters(), lr=0.01, max_iter=100)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.scaling_layer(all_logits) 
            loss = self.loss_func(scaled_logits, all_targets) 
            loss.backward()
            return loss

        logging.info("Optimizing Dirichlet (Matrix) scaling parameters...")
        optimizer.step(closure)
        logging.info("Dirichlet scaling completed.")
        return self.scaling_layer

def apply_posthoc_calibration(model, num_classes, val_loader, method, Mu):
    """
    Factory function to select and apply a post-hoc calibration method.
    """
    if method == "temperature":
        logging.info("Applying Temperature Scaling...")
        calibrator = TemperatureScaling(model).cuda()
        calibrator.calibrate(val_loader) 
        return calibrator
    
    elif method == "dirichlet":
        logging.info("Applying Dirichlet Scaling...")
        calibrator = DirichletScaling(model, num_classes, Mu=Mu).cuda()
        calibrator.calibrate(val_loader) 
        return calibrator
    
    else:
        logging.info("No calibration applied. Returning original model.")
        model.eval()
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-Hoc Calibration Script')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model_best.pth')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the specific dataset (e.g., .../data/Brain_tumour_dataset)')
    parser.add_argument('--model_name', type=str, required=True, help='e.g., resnet18, resnet50')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for the model')
    parser.add_argument('--method', type=str, required=True, choices=['temperature', 'dirichlet'], help='Calibration method')
    parser.add_argument('--Mu', type=float, default=0.0, help='Mu for Dirichlet regularizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout (for model build)')
    
    args = parser.parse_args()
    model_save_pth = Path(args.checkpoint_path).parent  
    log_file = model_save_pth / f"calibration_{args.method}.log"  
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Applying {args.method} calibration on: {args.checkpoint_path}")
    logging.info(f"Using dataset: {args.dataset_path}")

    try:
        _, val_loader, test_loader = get_data_loaders(args.dataset_path, args.batch_size)
    except Exception as e:
        logging.error(f"Failed to load data loaders: {e}")
        exit(1)
    
    logging.info(f"Data loaders created. Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    model = build_model(
        model_name=args.model_name, 
        num_classes=args.num_classes, 
        dropout=args.dropout
    ).cuda()

    logging.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("Model weights loaded successfully!")

    calibrated_model = apply_posthoc_calibration(
        model, 
        args.num_classes, 
        val_loader,  # Use validation set to fit calibrator
        args.method,
        args.Mu
    )

    calibration_metrics = CalibrationMetrics(n_bins=15)
    logging.info("Evaluating calibrated model on TEST set...")
    outputs, labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = calibrated_model(inputs)
            outputs.append(logits)
            labels.append(targets)

    outputs = torch.cat(outputs).to('cuda') 
    labels = torch.cat(labels).to('cuda') 

    test_metrics = {
        "accuracy": 100 * (outputs.argmax(dim=1) == labels).float().mean().item(),
        "ece": calibration_metrics.expected_calibration_error(outputs, labels),
        "mce": calibration_metrics.max_calibration_error(outputs, labels),
        "ace": calibration_metrics.adaptive_calibration_error(outputs, labels),
        "auc": calibration_metrics.compute_auc(outputs, labels).item(),
        "f1_score": calibration_metrics.compute_f1(outputs, labels)
    }

    logging.info("Post-hoc calibration completed.")
    logging.info(f"Test Metrics after Calibration: {test_metrics}")

    metrics_save_path = model_save_pth / f"calibrated_{args.method}_metrics.json"
    save_metrics_json(test_metrics, metrics_save_path)
    logging.info(f"Metrics saved to {metrics_save_path}")

    logging.info("Generating Reliability Diagram...")
    fig = calibration_metrics.plot_reliability_diagram(outputs, labels)
    fig.savefig(model_save_pth / f"reliability_diagram_{args.method}.png")

    logging.info(f"Calibration artifacts saved in: {model_save_pth}")