# Setup Instructions for New Environment For Demo

## Prerequisites

We use tinyproxy in order to proxy HTTP requests from the robot to the client to allow for package installation etc. You can install tinyproxy using apt

```bash
sudo apt install tinyproxy
```

## Connection

1. Connect to the robot via SSH. We include an additional -R argument to open a port for proxying.

```bash
sudo systemctl enable --now tinyproxy

ssh -X \
    -o ExitOnForwardFailure=yes \
    -R 8080:127.0.0.1:8888 \
    unitree@192.168.123.164
```

## Environment Setup

1. Once connected to the robot, clone this repo to the robot.

```bash
git clone https://github.com/NAPPING-AI/robot-vision-safety.git
cd robot-vision-safety
```

2. We use UV to manage the packages used for this project. If you don't have UV already installed, you can do so like so. More detailed instructions are available through their (page)[https://docs.astral.sh/uv/getting-started/installation/#standalone-installer]

```bash
# With Curl
curl -x 127.0.0.1:18080 -LsSf https://astral.sh/uv/install.sh | sh

# With wget
export use_proxy=yes
export http_proxy=127.0.0.1:8080
export https_proxy=127.0.0.1:8080
wget -qO- https://astral.sh/uv/install.sh | sh
```

3. Create a new UV environment with extra flag to use system python environment so we can make use of pre-installed packages that come with Jetpack SDK.

```bash
uv venv .venv --no-managed-python
```

4. To make things simpler in the future, we edit the environment to include environment variables for proxying.

```bash
# Open activate script in editor
vim .venv/bin/activate
```

Anywhere in the file, insert the following

```
http_proxy=127.0.0.1:8080
https_proxy=127.0.0.1:8080
HTTP_PROXY=http://127.0.0.1:8080
HTTPS_PROXY=http://127.0.0.1:8080
```

5. Finally activate the venv

```bash
source .venv/bin/activate
```

6. Install project dependencies

```bash
uv pip install ultralytics
uv pip install lap
uv pip install numpy==1.23.5
uv pip install pillow==10.4.0
```

7. Install Nviida Jetpack specific wheels of torch and torchvision

```bash
uv pip install --no-deps --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
uv pip install --no-deps --no-cache pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.16.2+c6f3977-cp38-cp38-linux_aarch64.whl
```

8. Install unitree_python_sdk package

```bash
# Setup CycloneDDS env vars
export CYCLONEDDS_HOME="$HOME/cyclonedds_ws/install/cyclonedds"
export CMAKE_PREFIX_PATH="$CYCLONEDDS_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# Install cyclonedds package
uv pip install cyclonedds==0.10.2

# Install unitree_sdk
uv pip install -e .
```

9. Test to see if unitree_sdk installed correctly

```bash
uv run python - <<'PY'
import unitree_sdk2py
import cyclonedds
import numpy
import cv2

print("Unitree SDK2: OK")
print("CycloneDDS:", cyclonedds.__file__)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
PY
```

10. Check to see if project runs correctly

```bash
cd example/r1/vision
uv run tracking.py
```