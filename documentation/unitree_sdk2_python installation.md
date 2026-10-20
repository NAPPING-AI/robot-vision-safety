# Install Unitree SDK2 Python on the NX

## 0. Connect to the backpack 

On your computer, start a proxy to give the backpack internet connection. SSH into it afterwards.

```bash
sudo apt install tinyproxy
sudo systemctl enable --now tinyproxy

ssh -X \
    -o ExitOnForwardFailure=yes \
    -R 127.0.0.1:18080:127.0.0.1:8888 \
    unitree@192.168.123.164
```

Once in the backpack, run:

```bash
export http_proxy=http://127.0.0.1:18080
export https_proxy=http://127.0.0.1:18080
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
```

## 1. Download Unitree SDK2 Python

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
```

## 2. Create the application environment

```bash
python3 -m venv \
  --without-pip \
  --system-site-packages \
  .venv
```

## 3. Create a temporary installer

```bash
python3 -m venv --without-pip /tmp/unitree-installer
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
/tmp/unitree-installer/bin/python /tmp/get-pip.py
```

## 4. Install the SDK into the application environment

```bash
/tmp/unitree-installer/bin/python -m pip \
  --python "$PWD/.venv" \
  install -e "$PWD"
```

A successful installation ends with packages including:

```text
cyclonedds-0.10.2
unitree_sdk2py-1.0.1
```

## 5. Remove the temporary installer

```bash
rm -rf /tmp/unitree-installer /tmp/get-pip.py
```

## 6. Activate and verify

```bash
source .venv/bin/activate

python - <<'PY'
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