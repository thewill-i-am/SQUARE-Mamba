import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
  sys.path.insert(0, str(BASE_DIR))

from functions.util import Create_dataset, load_data, parse_noise_spec
from networks.SQUARE_Mamba import SQUARE_Mamba


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(test_gen, model):
  model.eval()
  prediction = []
  with torch.no_grad():
    for iteration, (Data) in enumerate(test_gen):
      Data = Data.to(device)
      spei = model.forward(Data)
      prediction.append(spei.cpu().squeeze(1).numpy())
  return np.stack(prediction)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate the SQUARE-Mamba model on the Pooncarie test set.')
  parser.add_argument(
    '--noise',
    default=None,
    help="Comma-separated list of quantum noise channels, e.g. 'depolarizing=0.02,phase_flip=0.01'.",
  )
  parser.add_argument(
    '--noise-device',
    default=None,
    help='Override the PennyLane device used for the quantum encoder.',
  )
  parser.add_argument(
    '--noise-shots',
    type=int,
    default=None,
    help='Set the number of shots for stochastic noise simulation.',
  )
  args = parser.parse_args()

  noise_config = parse_noise_spec(args.noise)
  if noise_config is not None:
    if args.noise_device is not None:
      noise_config.setdefault('device', args.noise_device)
    if args.noise_shots is not None:
      noise_config.setdefault('shots', args.noise_shots)

  model = SQUARE_Mamba(
    in_channel=105,
    noise_config=noise_config,
    quantum_device=args.noise_device,
    shots=args.noise_shots,
  )
  model = model.to(device)

  data_Pooncarie, gt_Pooncarie = load_data(1260, 1476)
  testing_data, testing_gt = Create_dataset(data_Pooncarie, gt_Pooncarie, num_sample=201)
  testloader = DataLoader(testing_data, batch_size=201, shuffle=False)

  checkpoint_dir = BASE_DIR / "checkpoint"
  folder_path = checkpoint_dir / "SQUARE_Mamba_final_20251001_192812.pkl"
  model.load_state_dict(torch.load(folder_path, map_location=device))

  prediction_temp = test(testloader, model)
  gt_test = testing_gt[8:201, 4]
  prediction = prediction_temp[:, 8:201]

  gt_csv = pd.DataFrame(gt_test.reshape(-1, 1))
  prediction_csv = pd.DataFrame(prediction.reshape(-1, 1))

  result_dir = BASE_DIR / "Result" / "SQUARE_Mamba"
  result_dir.mkdir(parents=True, exist_ok=True)
  gt_csv.to_csv(result_dir / "gt_Pooncarie.csv", header=None, index=False)
  prediction_csv.to_csv(result_dir / "prediction_Pooncarie.csv", header=None, index=False)
