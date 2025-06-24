#!/bin/bash

CURRENT_DIR=$(pwd)

install() {
  # Update the raspberry pi
  sudo apt-get update
  sudo apt-get upgrade -y

  # Install necessary compilers and libraries
  sudo apt-get install -y build-essential tk-dev libncurses5-dev libncursesw5-dev \
  libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev \
  libexpat1-dev liblzma-dev zlib1g-dev libffi-dev uuid-dev git

  # Install additional libraries required for mmpose
  sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

  # Install Python 3.8 if not already installed
  if ! command -v python3.8 &> /dev/null; then
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tgz
    tar -xzf Python-3.8.16.tgz
    cd Python-3.8.16
    ./configure --enable-optimizations --with-ensurepip=install
    make -j$(nproc)
    sudo make altinstall
    cd ..
    rm -rf Python-3.8.16 Python-3.8.16.tgz
  fi

  # Update pip to the latest version
  python3.8 -m pip install --upgrade pip setuptools wheel

  # Create /opt/gestur directory if it doesn't exist
  if [ ! -d /opt/gestur ]; then
    sudo mkdir -p /opt/gestur
  fi

  # Clone the gestur repository if no .git directory exists
  cd /opt/gestur
  if [ ! -d /opt/gestur/.git ]; then
    sudo rm -rf /opt/gestur/*
    git clone https://github.com/xavi-burgos99/gestur.git /opt/gestur
  else
    git pull origin main
  fi
  cd "$CURRENT_DIR"

  # Remove existing virtual environment if it exists and create a new one
  if [ -d /opt/gestur/.venv ]; then
    sudo rm -rf /opt/gestur/.venv
  fi
  sudo python3.8 -m venv /opt/gestur/.venv
  sudo chown -R $(whoami):$(whoami) /opt/gestur/.venv
  sudo chmod -R 755 /opt/gestur/.venv
  sudo /opt/gestur/.venv/bin/python -m pip install --upgrade pip setuptools wheel

  # Install pytorch, mmpose and other dependencies
  sudo /opt/gestur/.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  sudo /opt/gestur/.venv/bin/python -m pip install mmpose
  sudo /opt/gestur/.venv/bin/python -m pip install numpy opencv-python
  sudo /opt/gestur/.venv/bin/python -m pip install mmdet==2.22.0
  sudo /opt/gestur/.venv/bin/python -m pip install mmengine

  cd /tmp
  if [ -d mmcv ]; then
    sudo rm -rf mmcv
  fi
  sudo git clone https://github.com/open-mmlab/mmcv.git
  cd mmcv
  export MMCV_WITH_OPS=1
  export FORCE_CUDA=0
  sudo /opt/gestur/.venv/bin/python -m pip install .
  cd ..
  sudo rm -rf mmcv
  cd "$CURRENT_DIR"

  sudo sed -i 's/< digit_version(mm/<= digit_version(mm/g' /opt/gestur/.venv/lib/python3.8/site-packages/mmdet/__init__.py

  # Install openmim with mmengine, mmcv and mmdet
  #sudo /opt/gestur/.venv/bin/python -m pip install -U openmim
  #sudo /opt/gestur/.venv/bin/python -m mim install mmengine

}

uninstall() {
  # Remove /opt/gestur directory if it exists
  if [ -d /opt/gestur ]; then
    sudo rm -rf /opt/gestur
  fi
}

# Check if the script is run with root privileges
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Use 'sudo' to run it."
  exit 1
fi

# Check if the first argument is provided
case "$1" in
  install)
    install
    ;;
  uninstall)
    uninstall
    ;;
  *)
    echo "Usage: $0 {install|uninstall}"
    exit 1
    ;;
esac
