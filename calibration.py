import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.optim as optim
import torch.nn as nn
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from utilities.misc import mkdir_p, save_metrics_json
from utilities.__init__ import save_checkpoint
from argparsor import parse_args 
from utilities.data_loader import get_data_loaders, get_dataset_info
from models.build_model import build_model  # <-- Use the flexible model builder
from utilities.metrics import CalibrationMetrics

# =============================================================================
# 1. BASE CLASS (CLEANUP)
# Moved to top level and given a clearer name.
# =============================================================================
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
        # Base forward pass just gets logits
        return self.base_model(x)

# =============================================================================
# 2. TEMPERATURE SCALING (IMPROVED & SAFER)
# Inherits from base class and adds safety constraint for T.
# =============================================================================
class TemperatureScaling(PostHocCalibrator):
    def __init__(self, base_model):
        super(TemperatureScaling, self).__init__(base_model)
        # Initialize T to 1.5, a common starting point
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.T

    def calibrate(self, val_loader):
        """
        Optimizes the temperature T on the validation set using LBFGS.
        Uses the *fast* method (caching logits).
        """
        criterion = nn.CrossEntropyLoss()
        
        # 1. Get all logits and labels ONCE
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

        # 2. Optimize T on the cached logits/labels
        # max_iter=100 is a good default
        optimizer = optim.LBFGS([self.T], lr=0.01, max_iter=100)

        def closure():
            optimizer.zero_grad()
            
            # --- ROBUSTNESS FIX ---
            # Constrain T to be positive to avoid division by zero or negative.
            # We optimize on T, but clamp it after each step.
            # A small epsilon (1e-4) is fine.
            self.T.data.clamp_(min=1e-4) 
            
            scaled_logits = all_logits / self.T
            loss = criterion(scaled_logits, all_targets)
            loss.backward()
            return loss

        logging.info("Optimizing temperature T...")
        optimizer.step(closure)
        
        # --- ROBUSTNESS FIX ---
        # Final check to ensure T is positive
        self.T.data.clamp_(min=1e-4) 
        
        logging.info(f"Optimal Temperature: {self.T.item()}")
        return self.T.item()

# =============================================================================
# 3. DIRICHLET SCALING (CORRECTED)
# Inherits from base class and uses correct calibration logic.
# =============================================================================
class DirichletScaling(PostHocCalibrator):
    def __init__(self, base_model, num_classes, Mu=0.0):
        super(DirichletScaling, self).__init__(base_model)
        self.num_classes = num_classes
        self.scaling_layer = nn.Linear(num_classes, num_classes)
        # Mu is the bias-only regularization (Lambda for weights is not used in original paper)
        self.Mu = Mu 

    def forward(self, x):
        logits = self.base_model(x)
        # --- CRITICAL FIX ---
        # Return raw logits from the new scaling layer.
        # DO NOT apply log_softmax here.
        scaled_logits = self.scaling_layer(logits)
        return scaled_logits

    def regularizer(self):
        # Regularizer only for the bias term (b_loss), as in the original paper
        W, b = self.scaling_layer.parameters()
        b_loss = ((b**2).sum()) / self.num_classes
        return self.Mu * b_loss

    def loss_func(self, outputs, targets):
        # CrossEntropyLoss expects raw logits, which is correct
        criterion = nn.CrossEntropyLoss()
        return criterion(outputs, targets) + self.regularizer()

    def calibrate(self, val_loader, epochs=50, lr=0.001):
        """
        Fits the linear scaling layer on the validation set.
        """
        optimizer = optim.Adam(self.scaling_layer.parameters(), lr=lr)
        
        logging.info("Fitting Dirichlet scaling layer...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            # --- CRITICAL FIX ---
            # Train *only* on the validation loader
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                
                optimizer.zero_grad()
                scaled_outputs = self.forward(inputs)
                loss = self.loss_func(scaled_outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
          
            if (epoch + 1) % 10 == 0: # Log every 10 epochs
                 logging.info(f'Dirichlet Calib. Epoch {epoch+1}, Avg Loss: {epoch_loss/num_batches:.4f}')

        logging.info("Dirichlet scaling completed.")
        return self.scaling_layer

# =============================================================================
# 4. CALIBRATION FACTORY (CLEANUP)
# Signature is now cleaner (no train_loader).
# =============================================================================
def apply_posthoc_calibration(model, num_classes, val_loader, args):
    """
    Factory function to select and apply a post-hoc calibration method.
    
    Args:
        model: The trained base model.
        num_classes: Number of classes.
        val_loader: DataLoader for the validation set (used for fitting).
        args: Command-line arguments.
    """
    if args.calibration == "temperature":
        logging.info("Applying Temperature Scaling...")
        calibrator = TemperatureScaling(model).cuda()
        calibrator.calibrate(val_loader) # Fit T on validation data
        return calibrator
    
    elif args.calibration == "dirichlet":
        logging.info("Applying Dirichlet Scaling...")
        calibrator = DirichletScaling(model, num_classes, Mu=args.Mu).cuda()
        # Fit the layer on validation data
        calibrator.calibrate(val_loader) 
        return calibrator
    
    else:
        logging.info("No calibration applied. Returning original model.")
        model.eval() # Make sure model is in eval mode
        return model

# =============================================================================
# 5. MAIN SCRIPT
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    # --- CHECK FOR REQUIRED ARG ---
    # This script is useless without a model to load.
    if not args.load_checkpoint:
        print("Error: Please provide a path to a trained model using --load_checkpoint")
        exit(1)
        
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    dataset_info = get_dataset_info(args.dataset)
    dataset_names = [info['name'] for info in dataset_info]
    
    if args.dataset_name not in dataset_names:
        raise ValueError(f"Dataset {args.D_name} not found. Available: {dataset_names}")
    
    dataset_index = dataset_names.index(args.dataset_name)
    
    # We only need val and test loaders for post-hoc calibration
    # Note: The 'train' loader from get_data_loaders is the training split,
    # which we are NOT using here. 'val' is the hold-out set.
    _, val_loaders, test_loaders = get_data_loaders(args.dataset, args.batch_size)
    valloader = val_loaders[dataset_index]
    testloader = test_loaders[dataset_index]
    
    model_save_pth = f"{args.checkpoint}/{args.dataset_name}/posthoc_{args.model}_{current_time}"
    mkdir_p(model_save_pth)
    
    log_file = os.path.join(model_save_pth, "calibration.log")
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
    
    logging.info(f"Applying post-hoc calibration on dataset: {args.dataset_name}")
    logging.info(f"Using model architecture: {args.model}")

    # --- FIX: Build model architecture ---
    model = build_model(
        model_name=args.model, 
        num_classes=args.num_classes, 
        dropout=args.dropout
    ).cuda()

    # --- FIX: Load the trained checkpoint ---
    logging.info(f"Loading checkpoint from: {args.load_checkpoint}")
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("Model weights loaded successfully!")

    # --- CLEANUP: Pass valloader, not trainloader ---
    calibrated_model = apply_posthoc_calibration(
        model, 
        args.num_classes, 
        valloader,  # Use validation set to fit calibrator
        args
    )

    calibration_metrics = CalibrationMetrics()

    # Evaluate on the TEST set
    logging.info("Evaluating calibrated model on TEST set...")
    outputs, labels = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
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
        "auc": calibration_metrics.compute_auc(outputs, labels),
        "f1_score": calibration_metrics.compute_f1(outputs, labels)
    }

    logging.info("Post-hoc calibration completed.")
    logging.info(f"Test Metrics after Calibration: {test_metrics}")

    # --- IMPROVEMENT: Save metrics to JSON ---
    metrics_save_path = os.path.join(model_save_pth, "calibrated_test_metrics.json")
    save_metrics_json(test_metrics, metrics_save_path)
    logging.info(f"Metrics saved to {metrics_save_path}")

    # Save the *calibrated* model
    # Note: This saves the base model + the calibration layer (T or Linear)
    save_checkpoint(
        {'state_dict': calibrated_model.state_dict()}, 
        is_best=False, 
        checkpoint=model_save_pth
    )

    logging.info("Generating Reliability Diagram...")
    fig = calibration_metrics.plot_reliability_diagram(outputs, labels)
    fig.savefig(os.path.join(model_save_pth, "reliability_diagram.png"))

    logging.info(f"Calibration artifacts saved in: {model_save_pth}")