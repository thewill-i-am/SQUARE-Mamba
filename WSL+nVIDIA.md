# Installing WSL

En caso de querrer correrlo sin nVIDIA GPU se debe comentar en requirements.txt la linea de mamba-ssm e ignorar la sección de CUDA.

```bat
wsl --install Ubuntu-22.04
```

Ubuntu as default distro can be changed by:

```bat
wsl --set-default Ubuntu-22.04
```

## Packages

```sh
sudo apt update && sudo apt upgrade -y
```

```sh
sudo apt install -y build-essential wget zip unzip nano python3 python3-venv python3-pip python3-dev git libopenblas-dev
```

## CUDA on WSL

Follow the instructions on [NVIDIA's website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-wsl-ubuntu-13-0-local_13.0.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-0-local_13.0.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

## Python environment

```sh
python3.10 -m venv .venv-3.10
source .venv-3.10/bin/activate
pip install --upgrade pip
```
Install Requirements

```sh
pip install -r requirements.txt
```