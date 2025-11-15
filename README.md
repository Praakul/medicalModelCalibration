# Medical Image Classification and Calibration

This project provides a complete, end-to-end pipeline for training, evaluating, and calibrating deep learning models for medical image classification. The primary goal is to analyze and improve the reliability of model confidence scores, ensuring that a model's predicted confidence aligns with its actual accuracy.

This pipeline is built to be flexible, allowing for easy switching between different models, datasets, loss functions, and calibration methods to rigorously test and produce trustworthy models.

## 🚀 Key Features

- **Flexible Model Architecture**: Easily train and calibrate ResNet18, ResNet34, and ResNet50 using a unified model builder.

- **Multi-Dataset Support**: Natively handles multiple datasets. The dataloader is designed to find and manage datasets (e.g., Brain_tumour_dataset, Breast_ultrasound_dataset) from a common data/ directory.

- **Advanced Loss Functions**: Includes standard CrossEntropyLoss as well as advanced losses for overconfident models, such as FocalLoss and combined losses like NLL+MDCA and FL+MDCA.

- **Post-Hoc Calibration**: Implements state-of-the-art post-hoc calibration methods, including Temperature Scaling (with safety constraints) and Dirichlet Scaling.

- **Comprehensive Reliability Metrics**: Calculates all standard classification and calibration metrics:
  - **Classification**: Accuracy, F1-Score (Weighted), AUC (OvR Weighted)
  - **Calibration**: ECE (Expected Calibration Error), ACE (Adaptive Calibration Error), and MCE (Max Calibration Error).

- **Structured Logging**: Outputs all results to organized checkpoint folders, including detailed .log files, metric summaries in .json files, and publication-ready Reliability Diagrams.

## Project Structure
```
.
├── argparsor.py            # Manages all command-line arguments
├── calibration.py          # Script for post-hoc calibration (Temp. & Dirichlet)
├── checkpoints/            # Output directory for all trained models & logs
│   ├── Brain_tumour_dataset/
│   └── ...
├── data/                   # Root directory for all datasets
│   ├── Brain_tumour_dataset/
│   │   ├── train/
│   │   │   ├── class_0/
│   │   │   └── class_1/
│   │   └── test/
│   │       ├── class_0/
│   │       └── class_1/
│   └── Breast_ultrasound_dataset/
│       └── ...
├── models/
│   └── build_model.py      # Flexible builder for ResNet18/34/50
├── runners.py              # Contains the core 'train' and 'test' epoch loops
├── train.py                # Main script for training the model
├── requirements.txt        # Project dependencies
└── utilities/
    ├── data_loader.py      # Loads, splits, and augments data
    ├── eval.py             # (Helper for accuracy)
    ├── losses.py           # Defines Focal, MDCA, and Combined losses
    ├── metrics.py          # Defines ECE, ACE, MCE, AUC, F1
    └── misc.py             # Helpers like AverageMeter and save_metrics_json
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/praakul/medicalModelCalibration
cd medicalModelCalibration
```

2. Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies: Your requirements.txt should include torch, torchvision, scikit-learn, matplotlib, numpy, and pillow.
```bash
pip install -r requirements.txt
```

4. **Data Setup**: This project requires a specific data structure. Place all your datasets (e.g., Brain_tumour_dataset) inside the data/ directory. Each dataset must contain train and test folders, which in turn contain class-specific subfolders.

   See the Project Structure section for the required layout.

## 🔬 How to Run

This project has a two-step workflow: (1) Training and (2) Calibration.

### Step 1: Training a Model

Use the `train.py` script to train your model. The script will automatically use your validation set (10% split from train) to find the best model, save its weights, and log its performance on the unseen test set.

**Example Command:**

This command trains a ResNet-34 on the Brain Tumour dataset (which has 4 classes) using the Focal Loss + MDCA combination.
```bash
python train.py \
    --model resnet34 \
    --dataset "A:/Model_Calibration/Code_Medical/" \
    --dataset_name "Brain_tumour_dataset" \
    --num_classes 4 \
    --loss "FL+MDCA" \
    --beta 5.0 \
    --gamma 2.0 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

**Key Training Arguments:**

- `--model`: The model to build (resnet18, resnet34, resnet50).
- `--dataset`: The root path to your project folder (which contains the data/ directory).
- `--dataset_name`: The specific dataset folder to use (e.g., Brain_tumour_dataset).
- `--num_classes`: The number of classes in your specific dataset.
- `--loss`: The loss function to use (cross_entropy, focal_loss, NLL+MDCA, FL+MDCA).
- `--epochs`: Total number of training epochs.
- `--beta` / `--gamma`: Parameters for the advanced loss functions.

### Step 2: Applying Post-Hoc Calibration

After training, use the `calibration.py` script to take your best-saved model and apply a post-hoc calibration method. This script will fit the calibrator on the validation set and then evaluate its final performance on the test set.

**Example Command:**

This command takes the best model from Step 1 (you must provide the path) and applies Temperature Scaling.
```bash
python calibration.py \
    --load_checkpoint "checkpoints/Brain_tumour_dataset/2025-11-15_19-30-45/model_best.pth" \
    --model resnet34 \
    --dataset "A:/Model_Calibration/Code_Medical/" \
    --dataset_name "Brain_tumour_dataset" \
    --num_classes 4 \
    --calibration temperature
```

**Key Calibration Arguments:**

- `--load_checkpoint`: (Required) Path to the model_best.pth file saved during training.
- `--calibration`: The method to apply (temperature or dirichlet).
- `--model`, `--dataset_name`, `--num_classes`: These must match the arguments used to train the checkpoint.

## 📊 Understanding the Output

All results are saved in the `checkpoints/` directory, organized by dataset and timestamp.

### Training Output

**Location:** `checkpoints/<dataset_name>/<timestamp>/`

- `model_best.pth`: The model weights (state_dict) from the epoch with the highest validation accuracy.
- `train.log`: A complete log of the training process, epoch by epoch.
- `best_test_metrics.json`: A JSON file containing the test set metrics (ECE, ACE, F1, etc.) for the uncalibrated best model.
- `reliability_diagram_best.png`: The reliability diagram of the uncalibrated best model.

### Calibration Output

**Location:** `checkpoints/<dataset_name>/posthoc_<model>_<timestamp>/`

- `model.pth`: The calibrated model's state_dict. This now includes the calibration layer (the learned T parameter or the Dirichlet scaling_layer).
- `calibration.log`: A log of the calibration process.
- `calibrated_test_metrics.json`: A JSON file with the final metrics (ECE, ACE, etc.) of the fully calibrated model.
- `reliability_diagram.png`: The new reliability diagram, showing the improved calibration.