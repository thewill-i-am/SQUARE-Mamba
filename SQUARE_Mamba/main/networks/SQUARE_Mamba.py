import os
import warnings
from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import (
  NoiseModel,
  amplitude_damping_error,
  depolarizing_error,
  pauli_error,
  phase_damping_error,
)
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

try:
  from mamba_ssm import Mamba  # type: ignore
except ImportError:  # pragma: no cover
  class Mamba(nn.Module):
    """
    Lightweight fallback implementation used when the optional mamba-ssm package is unavailable.
    Provides a simple depthwise temporal convolution to preserve tensor shapes.
    """

    def __init__(self, d_model: int, d_state=None, d_conv: int = 3, expand=None):
      super().__init__()
      padding = d_conv // 2
      self.net = nn.Sequential(
        nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=padding, groups=d_model),
        nn.GELU(),
        nn.Conv1d(d_model, d_model, kernel_size=1),
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.net(x.transpose(1, 2)).transpose(1, 2)


current_directory = os.getcwd()
file_path = f"{current_directory}"
os.chdir(file_path)


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

  ONE_QUBIT_GATES = ("rx", "ry", "rz", "sx")
  TWO_QUBIT_GATES = ("cx", "rxx")
  TARGET_QUBITS_1Q = (0, 1, 2)
  TARGET_QUBITS_2Q = ((0, 1), (1, 2))

  def __init__(self, config=None):
    self.device_name: Optional[str] = None
    self.shots: Optional[int] = None
    self.channels: list[dict[str, float]] = []
    self.noise_model: Optional[NoiseModel] = None

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

    if self.channels:
      self.noise_model = self._build_noise_model()

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
    return bool(self.channels)

  def _build_noise_model(self):
    noise_model = NoiseModel()
    for channel in self.channels:
      single_error, two_error = self._make_errors(channel)
      if single_error is not None:
        for gate in self.ONE_QUBIT_GATES:
          for qubit in self.TARGET_QUBITS_1Q:
            noise_model.add_quantum_error(single_error, gate, [qubit])
      if two_error is not None:
        for gate in self.TWO_QUBIT_GATES:
          for qubits in self.TARGET_QUBITS_2Q:
            noise_model.add_quantum_error(two_error, gate, list(qubits))
    return noise_model

  def _make_errors(self, channel):
    name = channel["name"]
    if name == "depolarizing":
      prob = channel["prob"]
      return depolarizing_error(prob, 1), depolarizing_error(prob, 2)

    if name == "amplitude_damping":
      gamma = channel["gamma"]
      single = amplitude_damping_error(gamma)
      return single, single.tensor(single)

    if name == "phase_damping":
      lam = channel["lam"]
      single = phase_damping_error(lam)
      return single, single.tensor(single)

    if name == "bit_flip":
      prob = channel["prob"]
      single = pauli_error([("X", prob), ("I", 1 - prob)])
      return single, single.tensor(single)

    if name == "phase_flip":
      prob = channel["prob"]
      single = pauli_error([("Z", prob), ("I", 1 - prob)])
      return single, single.tensor(single)

    return None, None

  def apply(self, _wires):
    # Retained for backwards compatibility; noise is injected via Aer noise models.
    return


def _resolve_method(backend_name: Optional[str]) -> Optional[str]:
  if backend_name is None:
    return None

  backend_name = backend_name.lower()
  alias_map = {
    "default.qubit": None,
    "default.qubit.autograd": None,
    "aer_simulator": None,
    "aer-simulator": None,
    "statevector": "statevector",
    "density_matrix": "density_matrix",
    "matrix_product_state": "matrix_product_state",
    "mps": "matrix_product_state",
  }

  if backend_name not in alias_map:
    warnings.warn(
      f"Backend '{backend_name}' is not recognized; default Aer simulator will be used.",
      RuntimeWarning,
    )
  return alias_map.get(backend_name)


def create_estimator(noise_manager: NoiseManager, override_device=None, shots=None) -> AerEstimator:
  estimator = AerEstimator()

  method = _resolve_method(override_device or noise_manager.device_name)
  if method is not None:
    estimator.options.method = method

  if noise_manager.noise_model is not None:
    estimator.options.noise_model = noise_manager.noise_model

  resolved_shots = shots if shots is not None else noise_manager.shots
  if resolved_shots is not None:
    estimator.options.default_shots = int(resolved_shots)

  return estimator


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
    x = x.transpose(1, 2)
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
    y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
    y = self.dropout(self.conv2(y).transpose(-1, 1))
    output = self.linear(self.norm(x + y))
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
    feature_local = feature_local.squeeze() + augmented_tensor[:, :, 1, 1]

    return feature_local.reshape(-1, 15, 7)


def _build_temporal_circuit(input_params: Iterable, weight_params: Iterable) -> QuantumCircuit:
  circuit = QuantumCircuit(3)

  for idx, param in enumerate(input_params):
    circuit.ry(param, idx)

  for idx in range(3):
    circuit.ry(weight_params[idx], idx)

  circuit.rxx(weight_params[9], 0, 1)

  for idx in range(3):
    circuit.rx(weight_params[3 + idx], idx)

  circuit.rxx(weight_params[10], 1, 2)

  for idx in range(3):
    circuit.ry(weight_params[6 + idx], idx)

  circuit.mcx([0, 1], 2, ctrl_state="10")
  circuit.mcx([1, 2], 0, ctrl_state="10")
  circuit.mcx([2, 0], 1, ctrl_state="10")

  return circuit


class QLTEM(nn.Module):
  NUM_QUBITS = 3
  NUM_GROUPS = 5
  OBSERVABLES = [
    SparsePauliOp.from_list([("ZII", 1.0)]),
    SparsePauliOp.from_list([("IZI", 1.0)]),
    SparsePauliOp.from_list([("IIZ", 1.0)]),
  ]

  def __init__(self, noise_config=None, quantum_device=None, shots=None):
    super(QLTEM, self).__init__()

    self.noise_manager = NoiseManager(noise_config)
    self.estimator = create_estimator(self.noise_manager, override_device=quantum_device, shots=shots)

    self.temporal_blocks = nn.ModuleList([self._build_temporal_block() for _ in range(self.NUM_GROUPS)])

  def _build_temporal_block(self):
    input_params = ParameterVector("x", self.NUM_QUBITS)
    weight_params = ParameterVector("θ", 11)

    circuit = _build_temporal_circuit(input_params, weight_params)

    qnn = EstimatorQNN(
      circuit=circuit,
      observables=self.OBSERVABLES,
      input_params=list(input_params),
      weight_params=list(weight_params),
      estimator=self.estimator,
      input_gradients=True,
    )

    initial_weights = torch.zeros(len(weight_params), dtype=torch.float32)
    return TorchConnector(qnn, initial_weights=initial_weights.clone().detach())

  def _apply_block(self, block, x):
    batch_size, time_step, feature = x.shape
    reshaped = rearrange(x, "b t f -> (b f) t", b=batch_size, f=feature)
    result = block(reshaped)
    result = result.to(device=x.device, dtype=x.dtype)
    result = rearrange(result, "(b f) t -> b t f", b=batch_size, f=feature)
    return result

  def forward(self, x):
    q_slices = [
      x[:, [0, 1, 2], :],
      x[:, [3, 4, 5], :],
      x[:, [6, 7, 8], :],
      x[:, [9, 10, 11], :],
      x[:, [12, 13, 14], :],
    ]

    outputs = [self._apply_block(block, tensor) for block, tensor in zip(self.temporal_blocks, q_slices)]
    return tuple(outputs)


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
    output_4 = self.conv4(self.mamba_4(group_4))

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

    ST_feature_one = ltem_1 + qltem_1
    ST_feature_two = ltem_2 + qltem_2
    ST_feature_three = ltem_3 + qltem_3
    ST_feature_four = ltem_4 + qltem_4
    ST_feature_five = ltem_5 + qltem_5
    spei = 3 * self.tanh(self.FFB(ST_feature_one, ST_feature_two, ST_feature_three, ST_feature_four, ST_feature_five))
    return spei


def run_two_qubit_qnn_example():
  """
  Minimal example demonstrating a 2-qubit Qiskit EstimatorQNN wrapped for PyTorch training.
  """
  import torch.nn.functional as F

  data_params = ParameterVector("φ", 2)
  weight_params = ParameterVector("θ", 2)
  circuit = QuantumCircuit(2)

  for idx, param in enumerate(data_params):
    circuit.ry(param, idx)

  circuit.cz(0, 1)

  for idx, param in enumerate(weight_params):
    circuit.rx(param, idx)

  observables = [
    SparsePauliOp.from_list([("ZI", 1.0)]),
    SparsePauliOp.from_list([("IZ", 1.0)]),
  ]

  estimator = AerEstimator()
  qnn = EstimatorQNN(
    circuit=circuit,
    observables=observables,
    input_params=list(data_params),
    weight_params=list(weight_params),
    estimator=estimator,
    input_gradients=True,
  )

  initial_weights = torch.zeros(len(weight_params), dtype=torch.float32)
  q_layer = TorchConnector(qnn, initial_weights=initial_weights)

  class SimpleHybridModel(nn.Module):
    def __init__(self, quantum_layer):
      super().__init__()
      self.quantum_layer = quantum_layer
      self.post = nn.Linear(2, 1)

    def forward(self, x):
      quantum_features = self.quantum_layer(x)
      activated = F.elu(quantum_features)
      return self.post(activated)

  model = SimpleHybridModel(q_layer)
  example_input = torch.tensor([[0.1, -0.2], [0.6, 0.8]], dtype=torch.float32)
  output = model(example_input)
  return output


if __name__ == "__main__":
  print("Example QNN output:", run_two_qubit_qnn_example())
