import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent.parent
# Directorio por defecto
_DEFAULT_DATA_DIR = _BASE_DIR / "CRU_data_montevideo"


def _read(feature: str, start_point: int, end_point: int, data_dir: Path) -> np.ndarray:
  path = data_dir / f"{feature}.csv"
  df = pd.read_csv(path, header=None)
  return df.iloc[start_point:end_point, :9].values.astype('float32')

def load_data(start_point, end_point, data_dir=None):
  """
  Carga datos climáticos desde el directorio especificado.
  
  Args:
    start_point: Punto de inicio temporal
    end_point: Punto final temporal  
    data_dir: Directorio de datos (str o Path). Si es None, usa el directorio por defecto.
  """
  if data_dir is None:
    data_dir = _DEFAULT_DATA_DIR
  else:
    data_dir = Path(data_dir)
  
  if not data_dir.exists():
    raise FileNotFoundError(f"El directorio de datos no existe: {data_dir}")
  
  cld = _read("cld", start_point, end_point, data_dir)
  tmn = _read("tmn", start_point, end_point, data_dir)
  tmp = _read("tmp", start_point, end_point, data_dir)
  tmx = _read("tmx", start_point, end_point, data_dir)
  vap = _read("vap", start_point, end_point, data_dir)
  pet = _read("pet", start_point, end_point, data_dir)
  pre = _read("pre", start_point, end_point, data_dir)
  GT = _read("spei", start_point, end_point, data_dir)
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
  X, gt = [], []
  
  for i in range(num_sample):  
    feature = data[i:i+15, :9, :7]
    for m in range(9):
      for n in range(7):
        scaler = StandardScaler()
        scaler.fit(feature[:, m, n].reshape(-1, 1))
        feature[:, m, n] = scaler.transform(feature[:, m, n].reshape(-1, 1)).reshape(-1)
    X.append(feature)
    gt.append(GT[i+15, :9])

  return torch.tensor(X).transpose(1, 2).float(), torch.tensor(gt).float()
