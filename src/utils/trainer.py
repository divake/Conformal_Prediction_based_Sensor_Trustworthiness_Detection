# src/utils/trainer.py
import torch
import torch.nn as nn
import logging
from pathlib import Path
from torch.amp import autocast, GradScaler  # Updated import

def train_model(model, train_loader, test_loader, num_epochs, 
                learning_rate, device='cuda', save_dir='checkpoints'):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Updated GradScaler initialization
    scaler = GradScaler('cuda')
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                          optimizer, device, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        
        scheduler.step()
    
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    return best_acc

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (points, targets) in enumerate(train_loader):
        points, targets = points.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Updated autocast
        with autocast('cuda'):
            outputs = model(points)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss/len(train_loader), 100.*correct/total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, targets in test_loader:
            points, targets = points.to(device), targets.to(device)
            
            # Updated autocast
            with autocast('cuda'):
                outputs = model(points)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(test_loader), 100.*correct/total