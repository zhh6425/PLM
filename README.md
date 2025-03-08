# Point Linguist Model

> Anonymous Repo of paper _Object-centric Point Linguist Model for Point Cloud Segmentationg_
ICCV 2025 under review (Paper ID 2042)

# Preparing data

We use datasets including [ScanNet](http://www.scan-net.org/), [ScanRefer](https://github.com/daveredrum/ScanRefer), [ReferIt3D](https://referit3d.github.io/#dataset), and the [Multi3DRefer](https://aspis.cmpt.sfu.ca/projects/multi3drefer/data/multi3drefer_train_val.zip).

# Install

Install MinkowskiEngine:
```
# install MinkowskiEngine for Mask3D detector
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" # clone the repo to third_party
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
pip install hydra-core loguru albumentations open3d
```

# Run Training
```
bash scripts/train.sh
```
