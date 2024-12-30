# src/utils/trainer.py
import torch
import torch.nn as nn
import logging
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time

logger = logging.getLogger(__name__)

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, 
                weight_decay, device='cuda', save_dir='checkpoints', 
                grad_norm_clip=None, warmup_epochs=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    
    # Setup learning rate scheduler with warmup
    warmup_scheduler = LinearLR(optimizer, 
                              start_factor=0.1, 
                              end_factor=1.0, 
                              total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, 
                                      T_max=num_epochs-warmup_epochs, 
                                      eta_min=1e-5)
    scheduler = SequentialLR(optimizer, 
                           schedulers=[warmup_scheduler, main_scheduler], 
                           milestones=[warmup_epochs])
    
    scaler = GradScaler()
    best_acc = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, grad_norm_clip
        )
        
        # Validation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save checkpoint if best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, save_dir / 'best_model.pth')
        
        # Regular checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        epoch_time = time.time() - start_time
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Time: {epoch_time:.2f}s')
        logger.info(f'LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Best Acc: {best_acc:.2f}%')
    
    return best_acc

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_norm_clip=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (points, targets) in enumerate(train_loader):
        points, targets = points.to(device), targets.to(device)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(points)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        
        if grad_norm_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(f'Batch [{batch_idx+1}/{len(train_loader)}], '
                       f'Loss: {loss.item():.4f}, '
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
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(points)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(test_loader), 100.*correct/total