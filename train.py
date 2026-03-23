import torch
import torch.optim as optim
from model import VioLanguageModel
import math

def get_lr(step, total_steps=2000, max_lr=5e-4, min_lr=5e-5):
    if step > total_steps: return min_lr
    decay_ratio = (1 + math.cos(math.pi * step / total_steps)) / 2
    return min_lr + decay_ratio * (max_lr - min_lr)

def train_loop(model, train_loader, device, max_steps=2000):
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    model.train()
    
    step = 0
    while step < max_steps:
        for inputs, targets in train_loader:
            step += 1
            if step > max_steps: break
            
            lr = get_lr(step)
            for param_group in optimizer.param_groups: param_group['lr'] = lr
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                logits, loss = model(inputs, targets)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            if step % 100 == 0:
                print(f'Step {step}/{max_steps} | Loss: {loss.item():.4f}')