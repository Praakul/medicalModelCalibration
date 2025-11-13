import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

class CustomResNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout=0.5):
        super(CustomResNet, self).__init__()
        
        # 1. Load the correct base model and weights
        if model_name == 'resnet18':
            self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = 512
        elif model_name == 'resnet34':
            self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            in_features = 512
        elif model_name == 'resnet50':
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # 2. Freeze the base model and remove its head
        # (This is a different, more standard way than your original forward pass)
        for param in self.base_model.parameters():
            param.requires_grad = True # Keep them trainable
            
        self.base_model.fc = nn.Identity()

        # 3. Create your custom head, now with a dynamic input layer
        self.custom_head = nn.Sequential(
            # --- DYNAMIC INPUT LAYER ---
            nn.Conv2d(in_features, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 1. Pass through all base model layers
        # (This is simpler and less error-prone than calling layer1, layer2, etc.)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        # 2. Pass through the custom head
        x = self.custom_head(x)
        return x

def build_model(model_name: str, num_classes: int, dropout: float):
    """Factory function to build the model"""
    return CustomResNet(model_name, num_classes, dropout)