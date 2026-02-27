import os
import torch
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from typing import List, Tuple
from argparsor import parse_args

class MedicalImageDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize medical image dataset from a specific path
        
        Args:
            dataset_path (str): Path to dataset directory containing image classes
            transform (transforms.Compose, optional): Image transformations
        """
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

        class_dirs = sorted([d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))])
        
        for label, class_name in enumerate(class_dirs):
            class_path = os.path.join(dataset_path, class_name)
            img_list = os.listdir(class_path)
            if not img_list:
                print(f"Warning: No images found in {class_path}")
                continue  
                
            for img_name in img_list:
                img_path = os.path.join(class_path, img_name)
                if not Path(img_path).suffix.lower() in valid_extensions:
                    continue
                self.images.append(img_path)
                self.labels.append(label)
        
        self.transforms = transform or self._default_transforms()

    def _default_transforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = self.transforms(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return img, label


def get_data_loaders(dataset_path: str, batch_size: int) -> Tuple[DataLoader,DataLoader,DataLoader]:
    """
    Create train, validation, and test dataloaders for multiple medical datasets
    
    Args:
        base_path (str): Base directory containing datasets
        batch_size (int): Batch size for dataloaders
    
    Returns:
        Tuple of lists of train, validation, and test DataLoaders
    """
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(dataset_path,'train')
    test_path = os.path.join(dataset_path,'test')

    train_dataset_obj = MedicalImageDataset(train_path, transform=train_transforms)
    val_dataset_obj = MedicalImageDataset(train_path, transform=eval_transforms)
    test_dataset_obj = MedicalImageDataset(test_path, transform=eval_transforms)

    generator = torch.Generator().manual_seed(42)
    val_size = int(0.1 * len(train_dataset_obj))
    train_size = len(train_dataset_obj) - val_size

    indices = torch.randperm(len(train_dataset_obj), generator=generator)

    train_dataset = torch.utils.data.Subset(train_dataset_obj, indices[:train_size])
    val_dataset = torch.utils.data.Subset(val_dataset_obj, indices[train_size:])
    test_dataset = test_dataset_obj

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
   
def get_dataset_info(dataset_path: str) -> List[dict]:
    """
    Get information about a SINGLE dataset.
    
    Args:
        dataset_path (str): Path to the specific dataset directory
                             (e.g., /.../data/Breast_ultrasound_dataset)
    
    Returns:
        List containing a single dictionary of dataset information.
    """
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    dataset_name = os.path.basename(dataset_path.rstrip(os.sep))
    
    num_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    info = {
        'name': dataset_name,
        'num_classes': num_classes,
        'train_path': train_path,
        'test_path': test_path
    }
    
    return [info]




