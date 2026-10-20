# LAS2-S full-resolution depth installation guide (`544 x 448`)

These instructions install LAS2-S on the Unitree R1.

Configuration:

- input per camera: `1 x 3 x 448 x 544`
- LAS2-S internal maximum disparity: `192 px`
- application reliability threshold: the existing `MAX_DISPARITY = 96 px`
- ONNX opset: `16`
- NX runtime: TensorRT `8.5.2`, FP16

## 1. Export ONNX on the NVIDIA laptop

Clone and pin the official repository:

```bash
mkdir -p ~/r1_vision_models
cd ~/r1_vision_models

git clone https://github.com/TomTomTommi/LiteAnyStereo.git
cd LiteAnyStereo
git checkout 8c97bd4c4da3712c2ac60003a23201dfdb5935f4
```

Copy the supplied container definition into the repository and build it:

```bash
cp ~/Downloads/las2_test/Dockerfile.export .

sudo docker build \
  -t las2-export:torch26 \
  -f Dockerfile.export \
  .
```

Create a TensorRT-8.5-targeted copy of the official exporter:

```bash
cp export_onnx.py export_onnx_trt85.py
sed -i 's/opset_version=18/opset_version=16/' export_onnx_trt85.py
```

Download LAS2-S and export a fixed-shape ONNX model:

```bash
mkdir -p output_r1_448x544_d192_trt85

sudo docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  las2-export:torch26 \
  python export_onnx_trt85.py \
    --version las2 \
    --model_size s \
    --restore_ckpt checkpoints/LAS2_S.pth \
    --width 544 \
    --height 448 \
    --max_disp 192 \
    --output_name output_r1_448x544_d192_trt85/las2_s_r1_448x544_d192_opset16.onnx \
    --simplify

sudo chown -R "$USER:$USER" output_r1_448x544_d192_trt85
```

## 2. Copy the ONNX model to the NX

Run on the laptop:

```bash
ssh unitree@192.168.123.164 \
  'mkdir -p ~/r1_vision/lite_any_stereo_v2'

scp \
  output_r1_448x544_d192_trt85/las2_s_r1_448x544_d192_opset16.onnx \
  unitree@192.168.123.164:~/r1_vision/lite_any_stereo_v2/
```

## 3. Build the FP16 TensorRT engine on the NX

Run with other GPU-heavy programs stopped:

```bash
cd ~/r1_vision/lite_any_stereo_v2

sudo jetson_clocks --store \
  "$PWD/jetson_clocks_before_las2_full.conf"
sudo jetson_clocks

/usr/src/tensorrt/bin/trtexec \
  --onnx=las2_s_r1_448x544_d192_opset16.onnx \
  --saveEngine=las2_s_r1_448x544_d192_opset16_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --minTiming=1 \
  --avgTiming=8 \
  --timingCacheFile=las2_s_r1_448x544_d192_ws2048.cache \
  --buildOnly \
  2>&1 | tee build_las2_s_r1_448x544_d192_ws2048.log

sudo jetson_clocks --restore \
  "$PWD/jetson_clocks_before_las2_full.conf"
```

The engine is deliberately built on the NX because TensorRT plans are tied to
the target GPU and TensorRT version.

Source: [official Lite Any Stereo repository](https://github.com/TomTomTommi/LiteAnyStereo)
