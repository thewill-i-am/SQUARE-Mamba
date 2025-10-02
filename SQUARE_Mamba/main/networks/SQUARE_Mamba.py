import os
import sys
import warnings
from functools import partial

import torch
import torch.nn as nn
import pennylane as qml
from einops import rearrange
from mamba_ssm import Mamba

current_directory = os.getcwd()
file_path = f"{current_directory}"
os.chdir(file_path)
##
def make_model(in_channel=105, noise_config=None, quantum_device=None, shots=None):
  return SQUARE_Mamba(
    in_channel=in_channel,
    noise_config=noise_config,
    quantum_device=quantum_device,
    shots=shots,
  )


class NoiseManager:
  SUPPORTED_CHANNELS = {
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "bit_flip",
    "phase_flip",
  }

  def __init__(self, config=None):
    self.device_name = None
    self.shots = None
    self.channels = []
    if config is None:
      return

    if isinstance(config, dict) and "channels" in config:
      channels = config.get("channels", [])
      self.device_name = config.get("device")
      self.shots = config.get("shots")
    else:
      channels = config

    if isinstance(channels, dict):
      channels = [channels]

    for channel in channels or []:
      parsed = self._parse_channel(channel)
      if parsed is not None:
        self.channels.append(parsed)

  def _parse_channel(self, channel_config):
    if not isinstance(channel_config, dict):
      warnings.warn("Ignoring invalid noise channel configuration (expected dict).", RuntimeWarning)
      return None

    name = channel_config.get("name") or channel_config.get("type")
    if name is None:
      warnings.warn("Noise channel is missing a 'name'/'type' key; skipping.", RuntimeWarning)
      return None

    name = name.lower()
    if name not in self.SUPPORTED_CHANNELS:
      warnings.warn(f"Noise channel '{name}' is not supported; skipping.", RuntimeWarning)
      return None

    if name == "depolarizing":
      prob = float(channel_config.get("probability", channel_config.get("p", 0.0)))
      prob = min(max(prob, 0.0), 1.0)
      return {"name": name, "prob": prob}

    if name == "amplitude_damping":
      gamma = float(channel_config.get("gamma", channel_config.get("probability", channel_config.get("p", 0.0))))
      gamma = min(max(gamma, 0.0), 1.0)
      return {"name": name, "gamma": gamma}

    if name == "phase_damping":
      lam = float(channel_config.get("lam", channel_config.get("probability", channel_config.get("p", 0.0))))
      lam = min(max(lam, 0.0), 1.0)
      return {"name": name, "lam": lam}

    if name == "bit_flip":
      prob = float(channel_config.get("probability", channel_config.get("p", 0.0)))
      prob = min(max(prob, 0.0), 1.0)
      return {"name": name, "prob": prob}

    if name == "phase_flip":
      prob = float(channel_config.get("probability", channel_config.get("p", 0.0)))
      prob = min(max(prob, 0.0), 1.0)
      return {"name": name, "prob": prob}

    return None

  @property
  def enabled(self):
    return len(self.channels) > 0

  def apply(self, wires):
    if not self.enabled:
      return

    for channel in self.channels:
      name = channel["name"]
      if name == "depolarizing":
        for wire in wires:
          qml.DepolarizingChannel(channel["prob"], wires=wire)
      elif name == "amplitude_damping":
        for wire in wires:
          qml.AmplitudeDamping(channel["gamma"], wires=wire)
      elif name == "phase_damping":
        for wire in wires:
          qml.PhaseDamping(channel["lam"], wires=wire)
      elif name == "bit_flip":
        for wire in wires:
          qml.BitFlip(channel["prob"], wires=wire)
      elif name == "phase_flip":
        for wire in wires:
          qml.PhaseFlip(channel["prob"], wires=wire)


def create_quantum_device(noise_manager, override_device=None, shots=None):
  device_name = override_device or noise_manager.device_name
  if device_name is None:
    device_name = "default.mixed" if noise_manager.enabled else "default.qubit"

  device_kwargs = {"wires": 3}
  resolved_shots = shots if shots is not None else noise_manager.shots
  if resolved_shots is not None:
    device_kwargs["shots"] = resolved_shots

  return qml.device(device_name, **device_kwargs)

def map_generation(spei_tensor, channels):

  batch_size = spei_tensor.shape[0]
  pixel_vector = spei_tensor.reshape(batch_size, 9, -1)
  map = torch.zeros((batch_size, 3, 3, channels), device=spei_tensor.device, dtype=spei_tensor.dtype)

  map[:, 0, 0:3, :] = pixel_vector[:, 0:3, :]
  map[:, 1, 0:3, :] = pixel_vector[:, 3:6, :]
  map[:, 2, 0:3, :] = pixel_vector[:, 6:9, :]

  return map.permute(0, 3, 1, 2) 


def nearest_padding(input_tensor):
  
  batch, channel = input_tensor.shape[0], input_tensor.shape[1]
  padding_tensor = input_tensor
  mask = padding_tensor[0, 0] == 0 
  non_zero_idx = torch.nonzero(~mask, as_tuple=True)
  zero_idx = torch.nonzero(mask, as_tuple=True)
  distances = (non_zero_idx[0][None, :] - zero_idx[0][:, None]) ** 2 + (non_zero_idx[1][None, :] - zero_idx[1][:, None]) ** 2
  
  nearest_idx = distances.argmin(dim=1)
  padding_values = padding_tensor[:batch, :channel, non_zero_idx[0][nearest_idx], non_zero_idx[1][nearest_idx]]
  padding_tensor[:batch, :channel, zero_idx[0], zero_idx[1]] = padding_values

  return padding_tensor
         

class ConvLayer(nn.Module):
  def __init__(self, d_model):
    super(ConvLayer, self).__init__()
    self.Conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular")
    self.norm = nn.BatchNorm1d(d_model)
    self.activation = nn.ELU()

  def forward(self, x):
      
    x = self.norm(x.permute(0, 2, 1))
    x = self.Conv(x)
    x = self.activation(x)
    x = x.transpose(1,2)
    return x

class auxiliary_decoder(nn.Module):
  def __init__(self, d_model, time_step, dropout=0.2):
    super(auxiliary_decoder, self).__init__()

    self.d_ff = 128
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=self.d_ff, kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=d_model, kernel_size=1)
    self.norm = nn.BatchNorm1d(time_step)
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(d_model, 1)
    self.linear2 = nn.Linear(time_step, 1)
    self.activation = nn.GELU()

  def forward(self, x):
      
    y = x
    y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
    y = self.dropout(self.conv2(y).transpose(-1,1))
    output = self.linear(self.norm(x+y))
    output = (self.linear2(output.transpose(-1, 1))).transpose(-1, 1)

    return output.squeeze(1)

class spatial_encoding_block(nn.Module): 
  def __init__(self, in_channel):
    super(spatial_encoding_block, self).__init__()
    
    self.in_channel = in_channel
    self.depthwise_conv = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=2, groups=self.in_channel)
    self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
    self.max_pooling = nn.MaxPool2d(2)

  def forward(self, augmented_tensor_temp):
    
    augmented_tensor = nearest_padding(augmented_tensor_temp)
    feature_local = self.max_pooling(self.leakyReLU(self.depthwise_conv(augmented_tensor)))
    feature_local = feature_local.squeeze()+augmented_tensor[:, :, 1, 1]
          
    return feature_local.reshape(-1, 15, 7)
  
  
def qnn(embedding, p, cp, noise_manager=None):

  measure_set = [0, 1, 2]
  groups = [[0, 1, 2]]

  for ws in groups:

    qml.RY(embedding[:, ws[0]], wires = ws[0])
    qml.RY(embedding[:, ws[1]], wires = ws[1])
    qml.RY(embedding[:, ws[2]], wires = ws[2])

    qml.RY(p[0, ws[0]], wires = ws[0])
    qml.RY(p[0, ws[1]], wires = ws[1])
    qml.RY(p[0, ws[2]], wires = ws[2])

    qml.IsingXX(cp[ws[0]], wires = [ws[0], ws[1]])

    qml.RX(p[1, ws[0]], wires = ws[0])
    qml.RX(p[1, ws[1]], wires = ws[1])
    qml.RX(p[1, ws[2]], wires = ws[2])

    qml.IsingXX(cp[ws[1]], wires = [ws[1], ws[2]])

    qml.RY(p[2, ws[0]], wires = ws[0])
    qml.RY(p[2, ws[1]], wires = ws[1])
    qml.RY(p[2, ws[2]], wires = ws[2])

    qml.ctrl(qml.PauliX, control=[ws[0], ws[1]], control_values="10")(ws[2])
    qml.ctrl(qml.PauliX, control=[ws[1], ws[2]], control_values="10")(ws[0])
    qml.ctrl(qml.PauliX, control=[ws[2], ws[0]], control_values="10")(ws[1])

    if noise_manager is not None:
      noise_manager.apply(ws)

  exp_vals_z = [qml.expval(qml.PauliZ(w)) for w in measure_set]
  return exp_vals_z
        
class QLTEM(nn.Module):
  def __init__(self, noise_config=None, quantum_device=None, shots=None):
    super(QLTEM, self).__init__()

    self.noise_manager = NoiseManager(noise_config)
    self.dev = create_quantum_device(self.noise_manager, override_device=quantum_device, shots=shots)

    self.qtemporal_1 = self._build_qnode()
    self.qtemporal_2 = self._build_qnode()
    self.qtemporal_3 = self._build_qnode()
    self.qtemporal_4 = self._build_qnode()
    self.qtemporal_5 = self._build_qnode()

    self.temporal1_v1 = nn.Parameter(torch.randn((3,  3)) * torch.tensor(0), True)
    self.temporal1_v2 = nn.Parameter(torch.randn(( 2)) * torch.tensor(0), True)
    
    self.temporal2_v1 = nn.Parameter(torch.randn((3,  3)) * torch.tensor(0), True)
    self.temporal2_v2 = nn.Parameter(torch.randn(( 2)) * torch.tensor(0), True)
    
    self.temporal3_v1 = nn.Parameter(torch.randn((3,  3)) * torch.tensor(0), True)
    self.temporal3_v2 = nn.Parameter(torch.randn(( 2)) * torch.tensor(0), True)
    
    self.temporal4_v1 = nn.Parameter(torch.randn((3, 3)) * torch.tensor(0), True)
    self.temporal4_v2 = nn.Parameter(torch.randn(( 2)) * torch.tensor(0), True)

    self.temporal5_v1 = nn.Parameter(torch.randn((3, 3)) * torch.tensor(0), True)
    self.temporal5_v2 = nn.Parameter(torch.randn(( 2)) * torch.tensor(0), True)

  def _build_qnode(self):
    circuit = partial(qnn, noise_manager=self.noise_manager)
    return qml.QNode(circuit, self.dev, interface="torch", diff_method='best')
  
  def forward(self,x):

    q1=x[:,[0,1,2],:]
    q2=x[:,[3,4,5],:]
    q3=x[:,[6,7,8],:]
    q4=x[:,[9,10,11],:]
    q5=x[:,[12,13,14],:]
    batch_size, time_step, feature = q1.shape
    
    q1=rearrange(q1,'b t f -> (b f)t', b=batch_size, t=time_step, f=feature)
    q1 = self.qtemporal_1(q1, self.temporal1_v1, self.temporal1_v2)
    q1 = torch.cat(q1).to(x.device, dtype=x.dtype)
    q1=rearrange(q1,'(t b f )-> b t f ', b=batch_size, t=time_step, f=feature)
    
    q2=rearrange(q2,'b t f -> (b f)t', b=batch_size, t=time_step, f=feature)
    q2 = self.qtemporal_2(q2, self.temporal2_v1, self.temporal2_v2)
    q2 = torch.cat(q2).to(x.device, dtype=x.dtype)
    q2=rearrange(q2,'(t b f )-> b t f ', b=batch_size, t=time_step, f=feature)
    
    q3=rearrange(q3,'b t f -> (b f)t', b=batch_size, t=time_step, f=feature)
    q3 = self.qtemporal_3(q3, self.temporal3_v1, self.temporal3_v2)
    q3 = torch.cat(q3).to(x.device, dtype=x.dtype)
    q3=rearrange(q3,'(t b f )-> b t f ', b=batch_size, t=time_step, f=feature)
    
    q4=rearrange(q4,'b t f -> (b f)t', b=batch_size, t=time_step, f=feature)
    q4 = self.qtemporal_4(q4, self.temporal4_v1, self.temporal4_v2)
    q4 = torch.cat(q4).to(x.device, dtype=x.dtype)
    q4=rearrange(q4,'(t b f )-> b t f ', b=batch_size, t=time_step, f=feature)

    q5=rearrange(q5,'b t f -> (b f)t', b=batch_size, t=time_step, f=feature)
    q5 = self.qtemporal_5(q5, self.temporal5_v1, self.temporal5_v2)
    q5 = torch.cat(q5).to(x.device, dtype=x.dtype)
    q5=rearrange(q5,'(t b f )-> b t f ', b=batch_size, t=time_step, f=feature)
    
    return q1, q2, q3, q4, q5


class LTEM(nn.Module):
  def __init__(self):
    super(LTEM, self).__init__()

    self.mamba_1 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_2 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_3 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_4 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)
    self.mamba_5 = Mamba(d_model=7, d_state=32, d_conv=3, expand=20,)

    self.conv1 = ConvLayer(d_model=7)
    self.conv2 = ConvLayer(d_model=7)
    self.conv3 = ConvLayer(d_model=7)
    self.conv4 = ConvLayer(d_model=7)
    self.conv5 = ConvLayer(d_model=7)

  def forward(self, x):
    
    local_temporal_group = torch.chunk(x, 5, dim=1)

    group_1 = local_temporal_group[0]  
    output_1 = self.conv1(self.mamba_1(group_1)) 
    
    group_2 = local_temporal_group[1]
    output_2 = self.conv2(self.mamba_2(group_2)) 
    
    group_3 = local_temporal_group[2]
    output_3 = self.conv3(self.mamba_3(group_3)) 
    
    group_4 = local_temporal_group[3]
    output_4 = self.conv4(self.mamba_4(group_4) ) 

    group_5 = local_temporal_group[4]
    output_5 = self.conv5(self.mamba_5(group_5)) 
    
    return output_1, output_2, output_3, output_4, output_5 

class feature_fusion_block(nn.Module):
  def __init__(self):
    super(feature_fusion_block, self).__init__()
    
    self.PDM = auxiliary_decoder(d_model=7, time_step=15)

  def forward(self, group_1, group_2, group_3, group_4, group_5):
    
    decoder_input = torch.concat([group_1, group_2, group_3, group_4, group_5], dim=1)
    predicted_SPEI = self.PDM(decoder_input)
    
    return predicted_SPEI

class SQUARE_Mamba(nn.Module):
  def __init__(self, in_channel, noise_config=None, quantum_device=None, shots=None):
    super(SQUARE_Mamba, self).__init__()

    self.in_channel = in_channel
    self.SEB = spatial_encoding_block(self.in_channel)
    self.LTEM = LTEM()
    self.QLTEM = QLTEM(noise_config=noise_config, quantum_device=quantum_device, shots=shots)
    self.FFB = feature_fusion_block()
    self.tanh = nn.Tanh()

  def forward(self, x):
    
    augmented_tensor = map_generation(x, self.in_channel) 
    spatial_feature = self.SEB(augmented_tensor) 
    
    ltem_1, ltem_2, ltem_3, ltem_4, ltem_5 = self.LTEM(spatial_feature)
    qltem_1, qltem_2, qltem_3, qltem_4, qltem_5 = self.QLTEM(spatial_feature)

    ST_feature_one = ltem_1+qltem_1
    ST_feature_two = ltem_2+qltem_2
    ST_feature_three = ltem_3+qltem_3
    ST_feature_four = ltem_4+qltem_4
    ST_feature_five = ltem_5+qltem_5
    spei = 3*self.tanh(self.FFB(ST_feature_one, ST_feature_two, ST_feature_three, ST_feature_four, ST_feature_five))
    return spei 
