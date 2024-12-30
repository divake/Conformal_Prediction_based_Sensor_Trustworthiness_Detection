# src/utils/data_utils.py
from torch.utils.data import DataLoader
from data.dataset import ModelNet40Dataset  # Changed to absolute import

def create_dataloaders(root_dir, batch_size, num_workers):
    train_dataset = ModelNet40Dataset(root_dir, 'train')
    test_dataset = ModelNet40Dataset(root_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)
    
    return train_loader, test_loader, len(train_dataset.classes)