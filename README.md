# Point Linguist Model

> Offical Repo of paper _Segment Any Object via Bridged Large 3D-Language Model_

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

Install PointNet++
```
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

# Run Training
```
bash scripts/train.sh
```
