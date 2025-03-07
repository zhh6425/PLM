import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import logging
import timm

###  code for Uni3D point cloud encoder   ####

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    data_type = data.dtype
    data = data.float()
    fps_idx = pointnet2_utils.furthest_point_sample(data.contiguous(), number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data.to(dtype=data_type)

# def index_points(points, idx):
#     """
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points

# def fps(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         distance = torch.min(distance, dist)
#         farthest = torch.max(distance, -1)[1]
#     return index_points(xyz, centroids)


# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Uni3DPointcloudEncoder(nn.Module):
    def __init__(self, 
                 point_transformer, 
                 pc_encoder_dim=512,
                 pc_feat_dim=768,
                 embed_dim=1024,
                 group_size=64,
                 num_group=512,
                 patch_dropout=0.):
        super().__init__()
        from easydict import EasyDict
        self.trans_dim = pc_feat_dim
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.num_group = num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  pc_encoder_dim
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer

        self._delete_unuse_module()

    def _delete_unuse_module(self):
        if hasattr(self.visual, 'head'):
            del self.visual.head

    def batch_group_divide(self, point_clouds_list):
        centers = []
        features = []

        for point_clouds in point_clouds_list:
            pts = point_clouds[:, 3:].unsqueeze(0)
            colors = point_clouds[:, :3].unsqueeze(0)

            _, center, feature = self.group_divider(pts, colors)
            centers.append(center)
            features.append(feature)

        return torch.cat(centers, dim=0), torch.cat(features, dim=0)

    def forward(self, point_clouds_list):
        # divide the point cloud in the same form. This is important
        center, features = self.batch_group_divide(point_clouds_list)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)

        # ModuleList not support forward
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
        # x = self.visual.norm(x[:, 0, :])
        x = self.visual.norm(x[:, 1:, :])
        x = self.visual.fc_norm(x)

        x = self.trans2embed(x)
        
        return x
    

CK_DICT = {
    "tiny": {
        "eva_model": "eva02_tiny_patch14_224",
        "pc_ckpt": "pll/ckpt/uni3d/uni3dti/model.pt",
        "pc_ckpt_nolvis": "pll/ckpt/uni3dtinolvis/model.pt"
    },
    "small": {
        "eva_model": "eva02_small_patch14_224",
        "pc_ckpt": "pll/ckpt/uni3d/uni3ds/model.pt",
        "pc_ckpt_nolvis": "pll/ckpt/uni3dsnolvis/model.pt"
    },
    "base": {
        "eva_model": "eva02_base_patch14_448",
        "pc_ckpt": "pll/ckpt/uni3d/uni3db/model.pt",
        "pc_ckpt_nolvis": "pll/ckpt/uni3dbnolvis/model.pt"
    },
    "large": {
        "eva_model": "eva02_large_patch14_448",
        "pc_ckpt": "pll/ckpt/uni3d/uni3dl/model.pt",
        "pc_ckpt_nolvis": "pll/ckpt/uni3dlnolvis/model.pt"
    },
    "giant": {
        "eva_model": "eva_giant_patch14_560",
        "pc_ckpt": "pll/ckpt/uni3d/uni3dg/model.pt",
        "pc_ckpt_nolvis": "pll/ckpt/uni3dgnolvis/model.pt"
    },
}


def build_encoder(model_name, num_group=512, group_size=64):
    eva_ckpt = CK_DICT[model_name]["eva_model"]
    pc_ckpt = CK_DICT[model_name]["pc_ckpt"]
    checkpoint = torch.load(pc_ckpt, map_location='cpu')
    transformer = timm.create_model(eva_ckpt, drop_path_rate=0., pretrained=False)

    state_dict = {k.replace('point_encoder.', ''): v for k, v in checkpoint['module'].items() if k.startswith('point_encoder.')}
    # state_dict = {k: v for k, v in state_dict.items() if (not k.startswith('trans2embed')) and (not k.startswith('encoder2trans'))}

    encoder = Uni3DPointcloudEncoder(
        transformer, 
        pc_feat_dim=transformer.embed_dim,
        group_size=group_size,
        num_group=num_group,
        )
    encoder.load_state_dict(state_dict, strict=False)

    return encoder
    

