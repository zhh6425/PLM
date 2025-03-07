import os
import random
import numpy as np
import torch
import json

from llms import conversation as conversation_lib
from utils import (
    SCENE_QUESTION_LIST, 
    ANSWER_LIST,
    ANSWER_LIST_NO,
    ANSWER_LIST_WO,
    get_post_transform,
    DEFAULT_POINT_TOKEN,
    )

def get_meta_data(data_root, dataset_name, split):
    caption_root = os.path.join(data_root, "caption/")
    caption = json.load(open(os.path.join(caption_root, f"{dataset_name}_{split}.json")))

    caption_list = sorted(list(set([data["scene_id"] for data in caption])))
    caption_data = [data for data in caption if data["scene_id"] in caption_list]

    return caption_data, caption_list

def get_obj_id(caption_data):
    if "object_ids" in caption_data:
        object_id = caption_data["object_ids"]  # list
    else:
        object_id = []
    return object_id

def get_related_id(caption_data):
    if "related_id" in caption_data:
        anchor_id = caption_data["related_id"].get("anchor", [])
        if isinstance(anchor_id, str): 
            anchor_id = json.loads(anchor_id)
        distractor_id = caption_data["related_id"].get("distractor", [])
        if isinstance(distractor_id, str): 
            distractor_id = json.loads(distractor_id)
    else:
        anchor_id = []
        distractor_id = []
    return anchor_id, distractor_id

class SceneCapDataset(torch.utils.data.Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "segment200",
        "instance",
        "mask200",
    ]

    def __init__(
        self,
        base_dir,
        tokenizer,
        transform=None,
        precision: str = "fp32",
        scene_cap_data="scenecap",
        sampling_ratios="1",
        split="train",
    ):

        self.base_dir = base_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.precision = precision

        self.question_list = SCENE_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.answer_list_no = ANSWER_LIST_NO
        self.answer_list_wo = ANSWER_LIST_WO

        self.scene_cap_data = scene_cap_data.split("||")
        self.sampling_ratios = [float(r) for r in sampling_ratios.split(",")]
        if not len(self.sampling_ratios) == len(self.scene_cap_data):
            self.sampling_ratios = [1] * len(self.scene_cap_data)
        self.caption_data = []
        self.scene_data_path = [os.path.join(base_dir, "scannet"), os.path.join(base_dir, "scannetpp")]
        self.dataset_indices = []
        start_idx = 0  

        self.split = split
        for ds, r in zip(self.scene_cap_data, self.sampling_ratios):
            caption_data, _ = get_meta_data(base_dir, ds, split=split)
            caption_data = self._sample_data(caption_data, r)

            self.caption_data.extend([(start_idx + i, ds, data) for i, data in enumerate(caption_data)])
            self.dataset_indices.append((start_idx, start_idx + len(caption_data)))
            start_idx += len(caption_data) 

        if "train" not in self.split:
            self.voxelize, self.post_transform = get_post_transform()

    def __len__(self):
        return len(self.caption_data)
    
    def _sample_data(self, caption_data, r):
        if r == 1:
            return caption_data
        else:
            num_samples = int(len(caption_data)*r)
            sampled_indices = np.random.choice(len(caption_data), num_samples, replace=(r>1))
            return [caption_data[i] for i in sampled_indices]

    def __getitem__(self, idx):
        if "train" in self.split:
            return self.prepare_train_data(idx)
        else:
            return self.prepare_test_data(idx)

    def prepare_test_data(self, idx):
        _, ds, caption_data = self.caption_data[idx]
        scene_data_path = self.scene_data_path[1] if ds == "scannetpp" else self.scene_data_path[0]
        data_dict = self.get_data(scene_data_path, caption_data["scene_id"])
        scene_name = data_dict["name"]

        object_id = get_obj_id(caption_data)
        captions = caption_data["caption"].strip()

        data_dict = self.transform(data_dict)
        inst_label = torch.tensor(data_dict["instance"], dtype=torch.long)

        if False:  #self.voxelize is not None:
            data_part_list = self.voxelize(data_dict)
        else:
            data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = [data_dict]

        data_dict_list = [self.post_transform(data_dict_i) for data_dict_i in data_part_list]

        points_coord = torch.stack([data_dict_i["coord"] for data_dict_i in data_dict_list])
        points_color = torch.stack([data_dict_i["color"] for data_dict_i in data_dict_list])
        mask_proposal = torch.stack([data_dict_i["mask200"] for data_dict_i in data_dict_list])
        points_index = torch.stack([data_dict_i["index"] for data_dict_i in data_dict_list])

        # multi mask output
        masks = []
        for oid in object_id:
            mask = torch.isin(inst_label, torch.tensor(oid))
            masks.append(mask)
        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((1, *inst_label.shape), dtype=torch.bool)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        if not "train" in self.split:
            conv.system = ""
        i = 0
        while i < len(points_coord):
            conv.messages = []
            conv.append_message(
                conv.roles[0],
                DEFAULT_POINT_TOKEN + "\n" + f"Can you provide a brief description of this indoor scene?",
            )
            conv.append_message(
                conv.roles[1], captions,
            )
            conversations.append(conv.get_prompt())
            i += 1

        return (
            scene_name,
            points_coord,
            points_color,
            points_index,
            mask_proposal,
            conversations,
            masks,
            [],
            [],
            None,
            None,
            [],
        )

    def prepare_train_data(self, idx):
        _, ds, caption_data = self.caption_data[idx]
        scene_data_path = self.scene_data_path[1] if ds == "scannetpp" else self.scene_data_path[0]
        data_dict = self.get_data(scene_data_path, caption_data["scene_id"])
        scene_name = data_dict["name"]

        object_id = get_obj_id(caption_data)
        captions = caption_data["caption"].strip()
        
        data_dict = self.transform(data_dict)

        points_coord = data_dict["coord"]
        points_color = data_dict["color"]
        mask_proposal = data_dict["mask200"]
        inst_label = data_dict["instance"].long()

        ### Prompt ###
        question_template = random.choice(self.question_list)
        questions = [question_template]
        answers = [captions]
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        # multi mask output
        masks = []
        for oid in object_id:
            mask = torch.isin(inst_label, torch.tensor(oid))
            masks.append(mask)
        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((1, *inst_label.shape), dtype=torch.bool)

        return (
            scene_name,
            points_coord,
            points_color,
            None,
            mask_proposal,
            conversations,
            masks,
            [],
            [],
            questions,
            None,
            [],
        )
    
    def get_data(self, scene_data_path, scene_id):
        name = scene_id
        data_path = os.path.join(scene_data_path, scene_id)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["mask200"] = data_dict["mask200"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        elif "segment" in data_dict.keys():
            if "scannetpp" in scene_data_path:
                data_dict["segment"] = data_dict["segment"][:, 0]
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if "instance" in data_dict.keys():
            if "scannetpp" in scene_data_path:
                data_dict["instance"] = data_dict["instance"][:, 0]
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        return data_dict
