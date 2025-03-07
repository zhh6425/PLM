import hydra
import torch

import sys
sys.path.append("./pll")

from mask3d.models.mask3d import Mask3D
from mask3d.models.mask3d_clip import Mask3DClip
from mask3d.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

import hydra
from hydra import initialize, compose
import MinkowskiEngine as ME
import open3d as o3d

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)

        self.matcher = hydra.utils.instantiate(cfg.matcher)
        weight_dict = {
            "loss_ce": self.matcher.cost_class,
            "loss_mask": self.matcher.cost_mask,
            "loss_dice": self.matcher.cost_dice,
        }
        self.criterion = hydra.utils.instantiate(
            cfg.loss, matcher=self.matcher, weight_dict=weight_dict
        )

    def get_backbone_feats(self, x):
        return self.model.get_backbone_feats(x)

    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)
    

def get_model(checkpoint_path=None):
    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    # general
    cfg.general.checkpoint = checkpoint_path
    dataset_name = checkpoint_path.split('/')[-1].split('_')[0]

    # model
    if "scannet200" in dataset_name:
        cfg.general.num_targets = 201
        cfg.data.num_labels = 200
        cfg.model.num_queries = 150
    
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh


def prepare_data_tensor(points_list, colors_list, device):
    coordinates_batch = []
    features_batch = []
    unique_maps = []
    inverse_maps = []
    
    for i, (points, colors) in enumerate(zip(points_list, colors_list)):

        if points.ndim == 3: # 1, N, 3
            points = points.squeeze(0)
            colors = colors.squeeze(0)

        coords = torch.floor(points / 0.02)
        
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            coordinates=coords,
            features=colors,
            return_index=True,
            return_inverse=True,
        )
        
        sample_coordinates = coords[unique_map]
        coordinates_batch.append(sample_coordinates.int())
        sample_features = torch.cat([
            colors[unique_map], points[unique_map]   # raw_coordinates
        ], dim=-1)
        features_batch.append(sample_features.float())
        unique_maps.append(unique_map)
        inverse_maps.append(inverse_map)
    
    coordinates, features = ME.utils.sparse_collate(
        coords=coordinates_batch,
        feats=features_batch,
    )

    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features[:, :-3],
        device=device,
    )
    
    return data, features, unique_maps, inverse_maps
    