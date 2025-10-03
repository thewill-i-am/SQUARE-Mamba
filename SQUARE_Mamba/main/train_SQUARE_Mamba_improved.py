import argparse
import getpass
import os
import sys
from pathlib import Path

import numpy as np
from tomlkit import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from importlib import import_module
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.chdir(BASE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from functions.util import Create_dataset, load_data, r_square

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--start_epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument('--model', default='SQUARE_Mamba', type=str, help='Import which network')
parser.add_argument('--lr', default=1e-4, help='initial learning rate (reduced from 1e-3)')
parser.add_argument('--epochs', type=int, help='Override the default epoch count (251).')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
parser.add_argument("--data-dir", type=str, default=None,help="Directorio de datos (por defecto: CRU_data)")
training_settings = [{'nEpochs': 251, 'start_epoch': 1}]

def validate(val_gen, model, epoch, best_loss_r2, best_checkpoint_path):
    model.eval()
    val_loss_list, val_loss_list_r2 = [], []

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    username = getpass.getuser()
    checkpoint_name = f"SQUARE_Mamba_{timestamp}-{username}.pkl"

    with torch.no_grad():
        for iteration, (Data, gt) in tqdm(enumerate(val_gen)):
            gt = gt.to(device)
            Data = Data.to(device)
            spei = model.forward(Data)
            loss = loss_function(gt, spei)

            val_loss_list.append(loss.item())
            loss_r2 = r_square(gt, spei)
            val_loss_list_r2.append(loss_r2.item())
            
        # Calculate averages OUTSIDE the loop
        val_loss = np.array(val_loss_list).mean()
        val_loss_r2 = np.array(val_loss_list_r2).mean()
        
        print("===>Epoch{} Part: Validation loss is :{:.4f}".format(epoch, val_loss))
        print("R² actual: {:.6f}".format(val_loss_r2))
        print("R² histórico mejor: {:.6f}".format(best_loss_r2))
        
        # Save current model (always save for recovery)
        os.makedirs("./checkpoint", exist_ok=True)
        current_checkpoint = f"./checkpoint/{checkpoint_name}"
        torch.save(model.state_dict(), current_checkpoint)
        
        # Save best model only if it improves
        if best_loss_r2 < val_loss_r2:
            best_loss_r2 = val_loss_r2
            torch.save(model.state_dict(), best_checkpoint_path)
            print("Nuevo record de R²: {:.6f} - Best model saved".format(val_loss_r2))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = getattr(validate, 'epochs_without_improvement', 0) + 1
            
        # Store for early stopping check
        validate.epochs_without_improvement = epochs_without_improvement

    return val_loss, best_loss_r2, epochs_without_improvement

def train(train_gen, model, optimizer, epoch):
    model.train()
    train_loss_list = []

    for iteration, (Data, gt) in tqdm(enumerate(train_gen)):
        optimizer.zero_grad()
        gt = gt.to(device)
        Data = Data.to(device)
        spei = model.forward(Data)

        loss = loss_function(gt, spei)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        train_loss_list.append(loss.item())

    train_loss = np.array(train_loss_list).mean()
    print("===>Epoch{} Part: Avg training loss is :{:.4f}".format(epoch, train_loss))
    return train_loss

if __name__ == '__main__':
    opt = parser.parse_args()
    
    training_data, gt_training = load_data(0, 960, opt.data_dir)
    validation_data, gt_validation = load_data(960, 1260, opt.data_dir)
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=945)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=285)

    loss_function = nn.MSELoss()
    Loss_list, Val_Loss_list, best_loss_r2 = [], [], -999
    if opt.epochs is not None:
        training_settings[0]['nEpochs'] = opt.epochs
    Loss_list, Val_Loss_list, best_loss_r2 = [], [], -999
    
    opt = parser.parse_args()
    if opt.epochs is not None:
        training_settings[0]['nEpochs'] = opt.epochs
    
    Net = import_module('networks.' + opt.model)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    model = Net.make_model().to(device)
    num_epoch = training_settings[0]['nEpochs']

    # IMPROVED OPTIMIZER SETTINGS
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(opt.lr), 
        weight_decay=0.01,  # Increased from 0.0001
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # IMPROVED SCHEDULER - more conservative
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Monitor R² (higher is better)
        factor=0.5,  # Reduce LR by half
        patience=10,  # Wait 10 epochs before reducing
        verbose=True
    )
    
    print("===> Loading model {} and criterion".format(opt.model))
    print("===> Learning rate: {}, Weight decay: 0.01".format(opt.lr))

    opt.nEpochs = training_settings[0]['nEpochs']
    opt.start_epoch = training_settings[0]['start_epoch']
    
    # IMPROVED DATA LOADING - add shuffle for better generalization
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)), 
        batch_size=315, 
        shuffle=True  # Changed from False
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)), 
        batch_size=285, 
        shuffle=False
    )
    
    # Best model checkpoint path
    best_checkpoint_path = "./checkpoint/SQUARE_Mamba_montevideo_best.pkl"
    epochs_without_improvement = 0
    
    print("===> Starting training with early stopping (patience: {})".format(opt.patience))
    
    for epoch in range(opt.start_epoch, training_settings[0]["nEpochs"] + 1):
        trainloss = train(trainloader, model, optimizer, epoch)
        Loss_list.append(trainloss)
        
        if epoch % 1 == 0:
            valloss, best_loss_r2, epochs_without_improvement = validate(
                valloader, model, epoch, best_loss_r2, best_checkpoint_path
            )
            Val_Loss_list.append(valloss)
            
            # Update scheduler with R² value
            scheduler.step(best_loss_r2)
            
            # Early stopping check
            if epochs_without_improvement >= opt.patience:
                print(f"\nEarly stopping triggered after {opt.patience} epochs without improvement")
                print(f"Best R² achieved: {best_loss_r2:.6f}")
                print(f"Best model saved at: {best_checkpoint_path}")
                break
    
    print(f"\nTraining completed. Final best R²: {best_loss_r2:.6f}")
    print(f"Best model available at: {best_checkpoint_path}")