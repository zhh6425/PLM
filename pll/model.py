from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask3d import get_model, prepare_data_tensor
from modules.mask_decoder import MaskAttentionModule, Attention

from llms.llava_llama import LlavaForCausalLM, LlavaModel
# from llms.llava_qwen import LlavaForCausalLM, LlavaModel

# init for vision encoder & projectors
class PLLMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PLLMetaModel, self).__init__(config)

        self.config = config
        if hasattr(self.config, "train_mask_decoder") and (
            config.eval_only or not kwargs.get("from_different_models", False)):
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_llm_modules(self.config)
            self.initialize_pll_modules(self.config)
        else:
            self.config.train_mask_decoder = kwargs.get("train_mask_decoder", True)
            try:
                self.config.out_dim = kwargs["out_dim"]
            except:
                pass
            self.vision_pretrained = kwargs.get("vision_pretrained", None)

    def initialize_mask_model(self,):
        mask_model = get_model(self.vision_pretrained)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_model = mask_model.to(device)

    def initialize_llm_modules(self, config):
        # Projection layer
        self.hidden_dim = config.hidden_size
        self.attn_proj = nn.Linear(96, 128)
        self.attn_layer = Attention(128, 8)
        self.attn_norm = nn.LayerNorm(128)
        self.mlp2llm = nn.Sequential(
                nn.Linear(128, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        
    def initialize_pll_modules(self, config):
        self.num_class = 2
        self.num_mask_tokens = 16
        self.out_dim = config.out_dim
        self.mask_pos_emb = nn.Sequential(
            nn.Linear(3, self.out_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.out_dim, self.out_dim),
            )
        self.mask_decoder_proj = nn.ModuleList([
            nn.Linear(128, self.out_dim),
            nn.Linear(96, self.out_dim), 
            ])
        self.mask_decoder = MaskAttentionModule(
            embedding_dim=self.out_dim, 
            num_heads=8, depth=3, 
            num_mask_tokens=self.num_mask_tokens
            )
        self.mlp2ref = nn.Sequential(
                nn.Linear(self.hidden_dim, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim, self.out_dim),
            )
        self.mlp2feat = nn.Sequential(
                nn.Linear(96, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim, self.out_dim),
            )
        self.mask_iou_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim, self.num_class), # 0 for target
            )
        self.mask_box_head = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.out_dim, 6),
            )


# super from LLMs
class PLLModel(PLLMetaModel, LlavaModel):
    def __init__(self, config, **kwargs):
        super(PLLModel, self).__init__(config, **kwargs)
        self.config.use_cache = False
        self.config.tune_mm_mlp_adapter = True


class PLLForCausalLM(LlavaForCausalLM):
    def __init__(self, config, **kwargs):
        self.eval_only = kwargs.pop("eval_only", getattr(config, "eval_only", False))
        config.eval_only = self.eval_only

        if not getattr(config, "train_mask_decoder", False) or not config.eval_only:
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.iou_loss_weight = kwargs.pop("iou_loss_weight", None)
            self.l1_loss_weight = kwargs.pop("l1_loss_weight", None)
            self.box_loss_weight = kwargs.pop("box_loss_weight", None)

        self.seg_token_idx = kwargs.pop("seg_token_idx", None)
        self.tgt_token_idx = kwargs.pop("tgt_token_idx", None)

        super(LlavaForCausalLM, self).__init__(config)

        self.model = PLLModel(config, **kwargs)

        # Initialize weights and apply final processing config.vocab_size
        self.post_init()
   
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def pairwise_iou(self, masks1: torch.Tensor, masks2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        masks1 = torch.sigmoid(masks1)  # N, M1
        masks2 = torch.sigmoid(masks2)  # N, M2
        masks1 = (masks1 > 0.5).to(dtype=masks1.dtype)
        masks2 = (masks2 > 0.5).to(dtype=masks2.dtype)
        intersection = torch.einsum("ni,nj->ij", masks1, masks2)    # M1, M2
        union = masks1.sum(0)[:, None] + masks2.sum(0)[None, :] - intersection  # M1, M2
        iou = intersection / (union + eps)
        return iou

    def get_token_mask(self, input_ids, token_idx, visual_token_lens):

        seg_token_mask = input_ids[:, 1:] == token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        max_len = max(visual_token_lens)
        seg_token_mask = torch.stack([
            torch.cat([
                torch.zeros((visual_len,), dtype=torch.bool, device=seg_token_mask.device),
                seg_token_mask[i],
                torch.zeros((max_len - visual_len,), dtype=torch.bool, device=seg_token_mask.device)
                ], dim=0)
                for i, visual_len in enumerate(visual_token_lens)
                ], dim=0)

        return seg_token_mask

    def sample_target_masks(self, masks_list, related_masks_list=None, target_boxes_list=None):
        # generate a banlance target_masks
        num_mask_tokens = self.model.num_mask_tokens
        targets = []
        for i, gt_masks in enumerate(masks_list):
            rel_masks = related_masks_list[i] if related_masks_list is not None else []
            n_gt = len(gt_masks)
            n_rel = len(rel_masks)

            # ensure sample all gt
            selected = gt_masks[:num_mask_tokens]
            labels = torch.zeros(len(selected), dtype=torch.long)
            target_boxes = target_boxes_list[i][:len(selected)]

            if n_rel == 0: 
                targets.append({"masks": selected, "labels": labels, "bbox": target_boxes})
                continue

            # sample rel and cut
            remains = num_mask_tokens - len(selected)
            selected_rel = rel_masks[:remains]
            selected = torch.cat([selected, selected_rel], dim=0)
            labels = torch.cat([labels, torch.ones(len(selected_rel), dtype=torch.long)], dim=0)
            target_boxes = target_boxes_list[i][:len(selected)]
            targets.append({"masks": selected, "labels": labels, "bbox": target_boxes})

        return targets

    def model_forward(
        self,
        points_coord: torch.FloatTensor,
        points_color: torch.FloatTensor,
        mask_proposal: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        related_masks_list: List[torch.FloatTensor],
        target_boxes_list: List[torch.FloatTensor],
        inference: bool = False,
        precision: str = "bf16",
        task_mode: str = "single",
        **kwargs,
    ):
        batch_size = len(points_coord)

        assert batch_size == len(offset) - 1
        assert isinstance(points_coord, list)

        # prepare sparse data
        data, features, unique_map, inverse_map = prepare_data_tensor(
            points_coord, 
            points_color, 
            device=points_coord[0].device)
        
        # mask3d encoder
        with torch.no_grad(): 
            self.model.mask_model.eval()
            pred_outputs = self.model.mask_model(data, raw_coordinates=features[:, -3:])

        if precision == "bf16":
            data_type = torch.bfloat16
        elif precision == "fp16":
            data_type = torch.float16
        else:
            data_type = torch.float32
            
        for key, value in pred_outputs.items():
            if isinstance(value, list):
                pred_outputs[key] = [v.to(dtype=data_type) for v in value]
            elif isinstance(value, torch.Tensor):
                pred_outputs[key] = value.to(dtype=data_type)
            else:
                pass

        point_wise_features = pred_outputs['backbone_features']
        queries_embeddings = pred_outputs['queries_embeddings']
        queries_coords = pred_outputs['sampled_coords']

        # pre fusion for scene and query
        scene_feat = [self.model.attn_proj(feat) for feat in point_wise_features]
        attn_out = torch.cat([self.model.attn_layer(q=q, k=k, v=k) for q, k in zip(queries_embeddings, scene_feat)], dim=0)
        queries_embeddings = self.model.attn_norm(queries_embeddings + attn_out)

        # project to llm
        input_embeddings = self.model.mlp2llm(queries_embeddings)  # ffn
        output = super().forward(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
        output_hidden_states = output.hidden_states
        last_hidden_state = output_hidden_states[-1]
        model_output = output
        ce_loss = model_output.loss

        if task_mode == "zero":
            ce_loss = ce_loss * self.ce_loss_weight
            loss = ce_loss
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                }
    
        # hack for POINT_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        visual_token_lens = [pf.size(0) - 1 for pf in input_embeddings]
        seg_token_mask = self.get_token_mask(input_ids, self.seg_token_idx, visual_token_lens)
        
        ref_embeddings = last_hidden_state[seg_token_mask]  # seg token  b, 1, c
        ref_embeddings = self.model.mlp2ref(ref_embeddings).unsqueeze(1)

        # decode for multi mask & retrieval
        queries_pos = self.model.mask_pos_emb(queries_coords)
        queries_embeddings = self.model.mask_decoder_proj[0](queries_embeddings)
        scene_embeddings = [self.model.mask_decoder_proj[1](feat) for feat in point_wise_features]
        mask_tokens = self.model.mask_decoder(queries=queries_embeddings, keys=ref_embeddings, src=scene_embeddings, queries_pos=queries_pos)

        pred_boxes = self.model.mask_box_head(mask_tokens)

        attn_out = self.model.mask_decoder.last_attn(q=mask_tokens, k=ref_embeddings, v=ref_embeddings)
        mask_logits = self.model.mask_decoder.last_norm(mask_tokens + attn_out)
        mask_logits = self.model.mask_iou_head(mask_logits)

        output_masks = {
            "pred_logits": mask_logits, 
            "pred_boxes": pred_boxes,
            "pred_masks": []}
        
        # sample the target masks
        target = self.sample_target_masks(masks_list, related_masks_list, target_boxes_list)

        for i in range(batch_size):
            pc_feat = self.model.mlp2feat(point_wise_features[i])
            mask = torch.mm(pc_feat, mask_tokens[i].T)       # N, num_mask_tokens
            output_masks["pred_masks"].append(mask[inverse_map[i]])

        pred_masks = [
            mp.squeeze(0).to(dtype=data_type) if mp.ndim == 3 else mp.to(dtype=data_type)
            for mp in mask_proposal
            ]
        pred_masks = [
            mask[:, (mask > 0).sum(dim=0) >= 150] for mask in pred_masks
            ]

        filted_output_masks = []
        for i in range(batch_size):
            output_mask = output_masks["pred_masks"][i]
            softmax_logit = F.softmax(mask_logits[i], dim=-1)

            selected_mask = softmax_logit.argmax(-1) == 0

            if not selected_mask.any(): # non pred object
                filted_output_masks.append(
                    torch.zeros_like(output_mask[:, :1]).to(output_mask)
                    )
                continue
            else: 
                out_masks = output_mask[:, selected_mask].detach()
                filted_output_masks.append(
                    out_masks
                    )
        
        gt_masks = [gt_mask.any(dim=0, keepdim=True) for gt_mask in masks_list]

        if inference: 

            return {
                "out_logit": model_output,
                "pred_masks": filted_output_masks,
                "gt_masks": gt_masks,
                "mask_list": [mask for i, mask in enumerate(pred_masks)],
            }
        
        else:

            # mask loss
            mask_losses, match_indices = self.model.mask_model.criterion(
                output_masks, target, mask_type="masks"
                )

            ce_loss = self.ce_loss_weight * ce_loss
            bbox_l1_loss = self.l1_loss_weight * mask_losses["loss_l1"]
            bbox_giou_loss = self.box_loss_weight * mask_losses["loss_boxes"]
            mask_bce_loss = self.bce_loss_weight * mask_losses["loss_mask"]
            mask_dice_loss = self.dice_loss_weight * mask_losses["loss_dice"]
            iou_loss = self.iou_loss_weight * mask_losses["loss_ce"]

            loss = ce_loss + mask_bce_loss + mask_dice_loss + iou_loss + bbox_l1_loss + bbox_giou_loss

            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "iou_loss": iou_loss,
                "l1_loss": bbox_l1_loss,
                "bbox_loss": bbox_giou_loss,
                "pred_masks": filted_output_masks,
                "gt_masks": gt_masks,
            }