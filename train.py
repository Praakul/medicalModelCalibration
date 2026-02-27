import os
import torch
import torch.optim as optim
import logging
from time import localtime, strftime
from utilities.misc import mkdir_p, save_metrics_json
from utilities.__init__ import save_checkpoint, create_save_path, get_lr
from argparsor import parse_args 
from utilities.data_loader import get_data_loaders, get_dataset_info
from models.resnet import build_model
from utilities.losses import loss_dict
from runners import train, test
from utilities.metrics import CalibrationMetrics
from datetime import datetime

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    dataset_info_list = get_dataset_info(args.dataset)
    dataset_info = dataset_info_list[0]
    dataset_name = dataset_info['name'] 
    train_loaders, val_loaders, test_loaders = get_data_loaders(args.dataset, args.batch_size)

    model_save_pth = f"{args.checkpoint}/{args.dataset.strip("/")[-1]}/{current_time}"
    mkdir_p(model_save_pth)
    
    log_file = os.path.join(model_save_pth, "train.log")
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
    
    logging.info(f"Training on dataset: {args.dataset} , loss : {args.loss}")
    logging.info(f"Building model: {args.model}")
    logging.info(f"The number of classes from dataset_info is {dataset_info['num_classes']}")
    logging.info("--- Run Hyperparameters ---")
    logging.info(f"  Classes:      {dataset_info['num_classes']}")
    logging.info(f"  Epochs:       {args.epochs}")
    logging.info(f"  Batch Size:   {args.batch_size}")
    logging.info(f"  Initial LR:   {args.learning_rate}")
    logging.info(f"  Weight Decay: {args.weight_decay}")
    
    if "focal" in args.loss.lower() or "fl" in args.loss.lower():
         logging.info(f"  Focal Gamma:  {args.gamma}")
    if "mdca" in args.loss.lower():
         logging.info(f"  MDCA Beta:    {args.beta}")
    logging.info("---------------------------")

    model = build_model(
        model_name=args.model,
        num_classes=dataset_info['num_classes'],
        dropout=args.dropout).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_gamma)
    criterion = loss_dict[args.loss](gamma=args.gamma, beta=args.beta)  
    calibration_metrics = CalibrationMetrics()  
    best_acc = 0.
    best_metrics = {}
    
    for epoch in range(0, args.epochs):
        logging.info(f"Epoch: [{epoch + 1} | {args.epochs}] LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train(train_loaders, model, optimizer, criterion)
        
        outputs, labels = test(val_loaders, model, criterion)  
    
        val_metrics = {
            "accuracy": 100*(outputs.argmax(dim=1) == labels).float().mean().item(),
            "ece": calibration_metrics.expected_calibration_error(outputs, labels),
            "mce": calibration_metrics.max_calibration_error(outputs, labels),
            "ace": calibration_metrics.adaptive_calibration_error(outputs, labels),
            "auc": calibration_metrics.compute_auc(outputs, labels).item(),
            "f1_score": calibration_metrics.compute_f1(outputs, labels)
            
        }
        
        outputs, labels = test(test_loaders, model, criterion) # Swap model and testloader
        test_metrics = {
            "accuracy": 100*(outputs.argmax(dim=1) == labels).float().mean().item(),
            "ece": calibration_metrics.expected_calibration_error(outputs, labels),
            "mce": calibration_metrics.max_calibration_error(outputs, labels),
            "ace": calibration_metrics.adaptive_calibration_error(outputs, labels),
            "auc": calibration_metrics.compute_auc(outputs, labels).item(),
            "f1_score": calibration_metrics.compute_f1(outputs, labels)
        }
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train_acc: {train_acc:.4f}, Val ECE: {val_metrics['ece']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
        logging.info(f"Val Metrics: {val_metrics}")
        logging.info(f"Test Metrics: {test_metrics}")
        
        is_best = val_metrics['accuracy'] > best_acc
        best_acc = max(best_acc, val_metrics['accuracy'])
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'dataset': args.dataset,
            'model': args.model
        }, is_best, checkpoint=model_save_pth)
        
        if is_best:
            best_metrics = test_metrics
            logging.info(f"New best model (Val Acc: {val_metrics['accuracy']:.4f})")
            logging.info(f"Corresponding Test Metrics: {test_metrics}")
    
    logging.info("Training completed...")
    logging.info("Best model metrics:")
    logging.info(best_metrics)

    metrics_save_path = os.path.join(model_save_pth, "Best_test_metrics.json")
    save_metrics_json(best_metrics, metrics_save_path)
        
    logging.info("Generating Reliability Diagram...")
    best_model_path = os.path.join(model_save_pth, "model_best.pth")
    checkpoint = torch.load(best_model_path,weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    outputs, labels = test(test_loaders, model, criterion)

    fig = calibration_metrics.plot_reliability_diagram(outputs, labels)
    reliability_plot_path = os.path.join(model_save_pth, "reliability_diagram_best.png")
    fig.savefig(reliability_plot_path)
   
    
    
