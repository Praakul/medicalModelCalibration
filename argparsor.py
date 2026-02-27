import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Medical Classification and Calibration Project')

    # Dataset Configuration
    parser.add_argument("--dataset", type=str, 
                       default="data/Breast_ultrasound_dataset",
                       help="Root path containing the data folder")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes in dataset")
    parser.add_argument('--data_workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--seed', default=100, type=int,
                        help='Random seed for reproducibility')
    
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Path to save model checkpoints")


    # Model Configuration
    parser.add_argument('--model', default='resnet18', 
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Base model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pre-trained weights')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout probability')

    # Training Configuration
    parser.add_argument('--epochs', default=50, type=int,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Training batch size')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Weight decay (L2 regularization)')

    # Loss Function Configuration
    parser.add_argument('--loss', type=str, default='cross_entropy', 
                        choices=['cross_entropy', 'focal_loss', 'NLL+MDCA', 'FL+MDCA'],
                        help='Loss function to use during training')
    parser.add_argument('--beta', default=5, type=float,
                        help='Weight of MDCA loss in combined loss (NLL+MDCA or FL+MDCA)')
    parser.add_argument('--gamma', default=2.0, type=float,
                        help='Gamma parameter for Focal Loss')

    # Calibration Configuration
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to the model_best.pth checkpoint to calibrate')
    
    parser.add_argument('--calibration', type=str, 
                        choices=['False', 'temperature', 'dirichlet'],
                        default='temperature',
                        help='Calibration methods to apply')

    # Evaluation Metrics
    parser.add_argument('--metrics', nargs='+', 
                        default=['accuracy', 'f1', 'ece', 'mce', 'auc', 'ace'],
                        choices=['accuracy', 'f1', 'ece', 'mce', 'auc', 'ace'],
                        help='Metrics to compute during evaluation')

    # Computational Resources
    parser.add_argument('--device', default='cuda', 
                        choices=['cuda', 'cpu'],
                        help='Computation device')

    # Logging and Checkpointing
    parser.add_argument('--results_dir', default='results', type=str,
                        help='Directory to store results')
    parser.add_argument('--checkpoint_interval', default=1, type=int,
                        help='Epochs between model checkpoints')

    # Learning Rate Scheduler Configuration
    parser.add_argument('--lr_scheduler', type=str, default='step', 
                        choices=['none', 'step', 'exponential', 'cosine'],
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--schedule_steps', default=[25 , 45] , type=int, 
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', default=0.1, type=float, 
                        help='Gamma parameter for learning rate scheduler')
    parser.add_argument('--Mu', default=.1, type=float, 
                         help='Mu for dirichlet scaling')

    return parser.parse_args()