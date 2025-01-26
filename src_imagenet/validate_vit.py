import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Configuration
IMAGENET_VAL_DIR = '/ssd_4TB/divake/CP_trust_IJCNN/dataset/imagenet/val'
CHECKPOINT_PATH = '/ssd_4TB/divake/CP_trust_IJCNN/checkpoints/vit_imagenet.pth'
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_val_transforms():
    """Get validation transforms for ImageNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_val_loader():
    """Create validation data loader."""
    val_dataset = datasets.ImageFolder(
        IMAGENET_VAL_DIR,
        transform=get_val_transforms()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    return val_loader, val_dataset.classes

def save_pretrained_model(model, save_path):
    """Save the pretrained model state dict."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_or_download_model():
    """Load model from checkpoint if exists, otherwise download pretrained."""
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    else:
        print("Downloading pretrained ViT model...")
        pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        save_pretrained_model(pretrained_model, CHECKPOINT_PATH)
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
    
    return model

def validate(model, val_loader):
    """Validate the model on ImageNet validation set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = outputs.max(1)
            
            # Update statistics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Print current accuracy
            current_acc = 100. * correct / total
            print(f"\rCurrent Accuracy: {current_acc:.2f}%", end='')
    
    final_accuracy = 100. * correct / total
    return final_accuracy

def main():
    print(f"Using device: {DEVICE}")
    
    # Load or download pretrained ViT model
    model = load_or_download_model()
    model = model.to(DEVICE)
    print("Model loaded successfully!")
    
    # Get validation loader
    print("Creating validation data loader...")
    val_loader, class_names = get_val_loader()
    print(f"Found {len(class_names)} classes")
    
    # Validate
    print("\nStarting validation...")
    accuracy = validate(model, val_loader)
    
    print(f"\n\nFinal Validation Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 