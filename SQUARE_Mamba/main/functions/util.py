import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_BASE_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _BASE_DIR / "CRU_data_montevideo"


def _read(feature: str, start_point: int, end_point: int) -> np.ndarray:
  path = _DATA_DIR / f"{feature}.csv"
  df = pd.read_csv(path, header=None)
  return df.iloc[start_point:end_point, :9].values.astype('float32')

def load_data(start_point, end_point):
  
  cld = _read("cld", start_point, end_point)
  tmn = _read("tmn", start_point, end_point)
  tmp = _read("tmp", start_point, end_point)
  tmx = _read("tmx", start_point, end_point)
  vap = _read("vap", start_point, end_point)
  pet = _read("pet", start_point, end_point)
  pre = _read("pre", start_point, end_point)
  GT = _read("spei", start_point, end_point)
  data = np.concatenate((cld.reshape(-1, 9, 1), tmn.reshape(-1, 9, 1), tmp.reshape(-1, 9, 1), tmx.reshape(-1, 9, 1), vap.reshape(-1, 9, 1), pet.reshape(-1, 9, 1), pre.reshape(-1, 9, 1)), axis = 2)

  return data, GT
      
def r_square(y_true, y_pred):
  y_true = y_true.view(-1)
  y_pred = y_pred.view(-1)
  ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
  ss_residual = torch.sum((y_true - y_pred) ** 2)
  r2 = 1 - (ss_residual / ss_total)
  return r2

def Create_dataset(data, GT, num_sample):
  features = np.empty((num_sample, 15, 9, 7), dtype=np.float32)
  targets = np.empty((num_sample, 9), dtype=np.float32)

  for i in range(num_sample):
    feature = data[i:i + 15, :9, :7].astype(np.float32, copy=False)
    mean = feature.mean(axis=0, keepdims=True)
    std = feature.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features[i] = (feature - mean) / std
    targets[i] = GT[i + 15, :9].astype(np.float32, copy=False)

  features = torch.from_numpy(features.swapaxes(1, 2))
  targets = torch.from_numpy(targets)
  return features, targets


def parse_noise_spec(spec: str | None):
  if spec is None:
    return None

  spec = spec.strip()
  if not spec or spec.lower() in {"none", "off", "false"}:
    return None

  channel_mappings = {
    "depolarizing": "probability",
    "bit_flip": "probability",
    "phase_flip": "probability",
    "phase_damping": "lam",
    "amplitude_damping": "gamma",
  }

  channels = []
  for chunk in spec.split(','):
    entry = chunk.strip()
    if not entry:
      continue
    if '=' not in entry:
      warnings.warn(
        f"Noise specification '{entry}' is invalid (expected format name=value); skipping.",
        RuntimeWarning,
      )
      continue

    name, value = entry.split('=', 1)
    channel_name = name.strip().lower()

    try:
      parameter_value = float(value)
    except ValueError:
      warnings.warn(
        f"Noise specification '{entry}' has a non-numeric value; skipping.",
        RuntimeWarning,
      )
      continue

    parameter_name = channel_mappings.get(channel_name, "probability")
    channel = {"name": channel_name, parameter_name: parameter_value}
    channels.append(channel)

  if not channels:
    return None

  return {"channels": channels}
