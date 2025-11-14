import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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

from functions.util import Create_dataset, load_data, parse_noise_spec, r_square

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--start_epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument('--model', default='SQUARE_Mamba', type=str, help='Import which network')
parser.add_argument('--lr', default=1e-3, help='initial learning rate')
parser.add_argument('--epochs', type=int, help='Override the default epoch count (251).')
parser.add_argument(
    '--noise',
    default=None,
    help="Comma-separated list of quantum noise channels, e.g. 'depolarizing=0.02,amplitude_damping=0.05'.",
)
parser.add_argument(
    '--noise-device',
    default=None,
    help="Override the PennyLane device used for the quantum encoder (defaults to default.qubit or default.mixed when noise is enabled).",
)
parser.add_argument(
    '--noise-shots',
    type=int,
    default=None,
    help='Set the number of shots for stochastic noise simulation (defaults to analytic expectation).',
)
training_settings = [{'nEpochs': 251, 'start_epoch': 1}]

def validate(val_gen, model, epoch, best_loss_r2):
    model.eval()
    val_loss_list, val_loss_list_r2 = [], []

    with torch.no_grad():
        for iteration, (Data, gt) in tqdm(enumerate(val_gen), total=len(val_gen), desc=f"[val] epoch {epoch}"):
            batch_start = time.time()
            gt = gt.to(device)
            Data = Data.to(device)
            transfer_time = time.time() - batch_start

            forward_start = time.time()
            spei = model.forward(Data)
            forward_time = time.time() - forward_start

            loss = loss_function(gt, spei)
            backward_time = 0.0  # no backward pass in eval

            val_loss_list.append(loss.item())
            loss_r2 = r_square(gt, spei)
            val_loss_list_r2.append(loss_r2.item())
            print(
                f"[val] epoch {epoch} batch {iteration+1}/{len(val_gen)} "
                f"loss={loss.item():.6f} r2={loss_r2.item():.6f} "
                f"(transfer {transfer_time:.3f}s, forward {forward_time:.3f}s)",
                flush=True,
            )

        # MOVER TODO ESTO FUERA DEL BUCLE FOR:
        val_loss = np.array(val_loss_list).mean()
        val_loss_r2 = np.array(val_loss_list_r2).mean()
        
        print("R² actual: {:.6f}".format(val_loss_r2))
        print("R² histórico mejor: {:.6f}".format(best_loss_r2))
        os.makedirs("./my_checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "./my_checkpoints/SQUARE_Mamba_new.pkl")
        print("Modelo guardado en epoca {}".format(epoch))

        if best_loss_r2 < val_loss_r2:
            best_loss_r2 = val_loss_r2
            print("Nuevo record de R²: {:.6f}".format(val_loss_r2))

    return val_loss, best_loss_r2
def train(train_gen, model, optimizer, epoch):
    model.train()
    train_loss_list = []

    for iteration, (Data, gt) in tqdm(enumerate(train_gen), total=len(train_gen), desc=f"[train] epoch {epoch}"):
        batch_start = time.time()
        optimizer.zero_grad()
        gt = gt.to(device)
        Data = Data.to(device)
        transfer_time = time.time() - batch_start

        forward_start = time.time()
        spei = model.forward(Data)
        forward_time = time.time() - forward_start

        loss = loss_function(gt, spei)
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start

        optimizer.step()
        train_loss_list.append(loss.item())
        train_loss = np.array(train_loss_list).mean()

        print(
            f"[train] epoch {epoch} batch {iteration+1}/{len(train_gen)} "
            f"loss={loss.item():.6f} avg_loss={train_loss:.6f} "
            f"(transfer {transfer_time:.3f}s, forward {forward_time:.3f}s, backward {backward_time:.3f}s)",
            flush=True,
        )

    return train_loss

if __name__ == '__main__':
    training_data, gt_training = load_data(0, 960)
    validation_data, gt_validation = load_data(960, 1260)
    train_samples = min(60, training_data.shape[0] - 15)
    val_samples = min(60, validation_data.shape[0] - 15)
    training_dataset, train_gt = Create_dataset(training_data, gt_training, num_sample=train_samples)
    validation_dataset, val_gt = Create_dataset(validation_data, gt_validation, num_sample=val_samples)

    loss_function = nn.MSELoss()
    Loss_list, Val_Loss_list, best_loss_r2 = [], [], -999
    opt = parser.parse_args()
    if opt.epochs is not None:
        training_settings[0]['nEpochs'] = opt.epochs
    Net = import_module('networks.' + opt.model)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    noise_config = parse_noise_spec(opt.noise)
    if noise_config is not None:
        if opt.noise_device is not None:
            noise_config.setdefault('device', opt.noise_device)
        if opt.noise_shots is not None:
            noise_config.setdefault('shots', opt.noise_shots)

    model = Net.make_model(
        noise_config=noise_config,
        quantum_device=opt.noise_device,
        shots=opt.noise_shots,
    ).to(device)
    num_epoch = training_settings[0]['nEpochs']

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch*3, eta_min=1e-7)
    print("===> Loading model {} and criterion".format(opt.model))

    opt.nEpochs = training_settings[0]['nEpochs']
    opt.start_epoch = training_settings[0]['start_epoch']
    trainloader = DataLoader(
        TensorDataset(training_dataset, train_gt[:, 4].reshape(-1, 1)),
        batch_size=min(64, len(training_dataset)),
        shuffle=False,
    )
    valloader = DataLoader(
        TensorDataset(validation_dataset, val_gt[:, 4].reshape(-1, 1)),
        batch_size=min(64, len(validation_dataset)),
        shuffle=False,
    )

    for epoch in range(opt.start_epoch, training_settings[0]["nEpochs"]):
        trainloss = train(trainloader, model, optimizer, epoch)
        Loss_list.append(trainloss.item())
        if epoch % 1 == 0:
            valloss, best_loss_r2 = validate(valloader, model, epoch, best_loss_r2)
            Val_Loss_list.append(valloss.item())
        scheduler.step()
