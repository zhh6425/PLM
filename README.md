<h1 align="center">Point Linguist Model: Segment Any Object via Bridged Large 3D-Language Model</h1>

<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/Repo-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/zhh6425/PLM/tree/main)
[![arXiv](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arXiv&logoColor=white&labelColor)](https://arxiv.org/abs/2509.07825)

</div>

---

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

## Citation

If you find this code useful for your research, please cite the following paper.

```bibtex
@misc{huang2026pointlinguistmodelsegment,
      title={Point Linguist Model: Segment Any Object via Bridged Large 3D-Language Model}, 
      author={Zhuoxu Huang and Mingqi Gao and Jungong Han},
      year={2026},
      eprint={2509.07825},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.07825}, 
}
```
