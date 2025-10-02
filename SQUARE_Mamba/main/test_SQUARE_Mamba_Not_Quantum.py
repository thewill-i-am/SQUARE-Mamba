import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
  sys.path.insert(0, str(BASE_DIR))

from functions.util import Create_dataset, load_data
from networks.SQUARE_Mamba_Not_Quantum import SQUARE_Mamba

# Parser de argumentos
parser = argparse.ArgumentParser(description="Test SQUARE_Mamba_Not_Quantum model")
parser.add_argument(
    "--checkpoint", 
    type=str, 
    required=True,
    help="Nombre del archivo checkpoint (.pkl) a cargar desde la carpeta checkpoint/"
)

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
  # Parsear argumentos
  args = parser.parse_args()
  
  model = SQUARE_Mamba(in_channel=105)
  model = model.to(device)

  data_Pooncarie, gt_Pooncarie = load_data(1260, 1476)
  testing_data, testing_gt = Create_dataset(data_Pooncarie, gt_Pooncarie, num_sample=201)
  testloader = DataLoader(testing_data, batch_size=201, shuffle=False)

  # Cargar checkpoint especificado por argumento
  checkpoint_dir = BASE_DIR / "checkpoint"
  checkpoint_path = checkpoint_dir / args.checkpoint
  
  # Verificar que el archivo existe
  if not checkpoint_path.exists():
    print(f"❌ Error: No se encontró el checkpoint: {checkpoint_path}")
    print(f"📁 Archivos disponibles en {checkpoint_dir}:")
    if checkpoint_dir.exists():
      for pkl_file in checkpoint_dir.glob("*.pkl"):
        print(f"   - {pkl_file.name}")
    else:
      print("   (El directorio checkpoint no existe)")
    sys.exit(1)
  
  print(f"📂 Cargando checkpoint: {args.checkpoint}")
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  print(f"✅ Modelo cargado exitosamente")

  prediction_temp = test(testloader, model)
  gt_test = testing_gt[8:201, 4]
  prediction = prediction_temp[:, 8:201]

  gt_csv = pd.DataFrame(gt_test.reshape(-1, 1))
  prediction_csv = pd.DataFrame(prediction.reshape(-1, 1))

  # Generar timestamp para el nombre de la carpeta
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  
  result_dir = BASE_DIR / "Result" / f"SQUARE_Mamba_Not_Quantum_{timestamp}"
  result_dir.mkdir(parents=True, exist_ok=True)
  
  # Guardar archivos con nombres originales
  gt_csv.to_csv(result_dir / "gt_Pooncarie.csv", header=None, index=False)
  prediction_csv.to_csv(result_dir / "prediction_Pooncarie.csv", header=None, index=False)
  
  print(f"✅ Resultados guardados (Not Quantum):")
  print(f"   Ground Truth: gt_Pooncarie.csv")
  print(f"   Predicción:   prediction_Pooncarie.csv")
  print(f"   Directorio:   {result_dir}")
