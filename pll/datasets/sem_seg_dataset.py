import glob
import json
import re
import os
import random
import numpy as np
import torch
from collections.abc import Sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

from llms import conversation as conversation_lib
from utils import (
    ANSWER_LIST,
    ANSWER_LIST_NO,
    ANSWER_LIST_WO,
    SHORT_QUESTION_LIST, 
    DEFAULT_POINT_TOKEN,
    get_post_transform,
    )

def get_meta_data(data_root, dataset_name, class_num='', split='train'):
    classes = []
    with open(os.path.join(data_root, f"metadata/{dataset_name}_classes{class_num}.txt")) as f:
        for line in f.readlines():
            classes.append(line.strip())

    with open(os.path.join(data_root, f"metadata/{dataset_name}_classes{class_num}.json"), "r", encoding="utf-8") as f:
        classes_dict = json.load(f)

    data_list = []
    if isinstance(split, str):
        with open(os.path.join(data_root, f"metadata/{dataset_name}_{split}.txt")) as f:
            for line in f.readlines():
                data_list.append(os.path.join(data_root, dataset_name, line.strip()))
    elif isinstance(split, Sequence):
        data_list = []
        for sp in split:
            data_list += glob.glob(os.path.join(data_root, dataset_name, sp, "*"))
    else:
        raise NotImplementedError                      

    return classes, classes_dict, data_list

class SemSegDataset(torch.utils.data.Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "instance",
        "mask200",
    ]

    def __init__(
        self,
        base_pc_dir,
        tokenizer,
        transform=None,
        precision: str = "fp32",
        sem_seg_data="scannetpp||scannet200||s3dis",
        split="train",
        vocab_list=None,
    ):
        self.base_pc_dir = base_pc_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.precision = precision

        self.question_list_sem = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.answer_list_no = ANSWER_LIST_NO
        self.answer_list_wo = ANSWER_LIST_WO

        self.sem_seg_datas = sem_seg_data.split("||")
        self.classes = {}
        self.data_list = []
        self.dataset_indices = []
        start_idx = 0  

        self.split = split
        self.vocab_list = vocab_list
        assert self.vocab_list is None or len(self.sem_seg_datas) <= 1, \
            "No nore then 1 dataset in testing."

        for ds in self.sem_seg_datas:
            if ("scannetpp" in ds) or ("replica" in ds):
                classes, classes_dict, data_list = get_meta_data(base_pc_dir, ds, split=split)
            elif "scannet" in ds:
                class_num = int(ds[7:])
                classes, classes_dict, data_list = get_meta_data(base_pc_dir, ds[:7], class_num=class_num, split=split)
                if class_num == 20:
                    classes = classes[:-1]
            elif "s3dis" in ds:
                data_sets = {"train": [f"Area_{i}" for i in [1, 2, 3, 4, 6]],
                             "val": ["Area_5"], "test": ["Area_5"],}
                classes, classes_dict, data_list = get_meta_data(base_pc_dir, ds, split=data_sets[split])
                classes = classes[:-1]
    
            self.classes[ds] = (classes, classes_dict)

            if self.vocab_list is not None:
                self.data_list.extend([
                    (start_idx + i, ds, path, vocab)
                    for i, path in enumerate(data_list)
                    for vocab in self.vocab_list
                    ])
            else:
                self.data_list.extend([
                    (start_idx + i, ds, path)
                    for i, path in enumerate(data_list)
                    # for vocab in classes
                    ])
            
            self.dataset_indices.append((start_idx, start_idx + len(data_list)))
            start_idx += len(data_list)    

        if "train" not in self.split:
            self.voxelize, self.post_transform = get_post_transform()

    def __len__(self):
        return len(self.data_list) 

    def __getitem__(self, idx):
        if "train" in self.split:
            return self.prepare_train_data(idx)
        else:
            return self.prepare_test_data(idx)
        
    def prepare_test_data(self, idx):

        _, ds, data_path, vocab = self.data_list[idx]
        vocab = re.sub(r'_', ' ', vocab)
    
        data_dict = self.get_data(data_path, ds)
        scene_name = data_dict["name"]

        data_dict = self.transform(data_dict)
        
        label = torch.tensor(data_dict["segment"], dtype=torch.long)
        unique_label = torch.unique(label).tolist()
        if -1 in unique_label:
            unique_label.remove(-1)
        if ds == "scannet20" and 19 in unique_label:
            unique_label.remove(19)
        if ds == "s3dis" and 12 in unique_label:
            unique_label.remove(12)

        inst_label = torch.tensor(data_dict["instance"], dtype=torch.long)
        unique_inst_label = torch.unique(inst_label).tolist()
        if -1 in unique_inst_label:
            unique_inst_label.remove(-1)

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

        bboxes = data_dict_list[0]["bbox"]

        classes_name, classes_dict = self.classes[ds]

        if not ds == "replica":
            classes = [classes_name[class_id] for class_id in unique_label]
            sampled_class = random.choice(classes) if vocab is None else vocab
        else:
            assert vocab is not None
            sampled_class = vocab
        assert sampled_class in classes_name, f"{sampled_class} not in {classes_name}"
        sem_id = classes_name.index(sampled_class)
        text = sampled_class

        distractor_classes = classes_dict[sampled_class]
        distractor_id = [classes_name.index(dc) for dc in distractor_classes]
       
        class_ids = [sem_id]
        sampled_classes = [sampled_class]

        mask = (label == sem_id)
        unique_inst_ids = inst_label[mask].unique()
        if -1 in unique_inst_ids:
            unique_inst_ids.remove(-1)
        masks = [(inst_label == inst_id) for inst_id in unique_inst_ids]
        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
            target_boxes = [bboxes.get(inst_id, torch.zeros(8)) for inst_id in unique_inst_ids]
        else:
            masks = mask.unsqueeze(0)
            target_boxes = [torch.zeros(8)]  

        distractor_mask = []
        distractor_boxes = []
        for id in distractor_id:
            d_mask = (label == id)
            distractor_unique_inst_ids = inst_label[d_mask].unique()
            distractor_mask += [(inst_label == inst_id) for inst_id in distractor_unique_inst_ids]
            distractor_boxes += [bboxes.get(inst_id, torch.zeros(8)) for inst_id in distractor_unique_inst_ids]

        related_masks = distractor_mask
        target_boxes = target_boxes + distractor_boxes

        if len(related_masks) > 0:
            related_masks = torch.stack(related_masks, dim=0)

        if len(target_boxes) > 0:
            target_boxes = torch.stack(target_boxes, dim=0)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        if not "train" in self.split:
            conv.system = ""
        i = 0
        while i < 1:  # one sample per time
            conv.messages = []
            conv.append_message(
                conv.roles[0],
                DEFAULT_POINT_TOKEN + "\n" + f"Please segment the {sampled_class} category in this point cloud.",
            )
            conv.append_message(
                conv.roles[1], f"It is [SEG]."
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
            related_masks,
            target_boxes,
            None,
            sampled_class,
            class_ids,
        )
    
    def prepare_train_data(self, idx):

        _, ds, data_path = self.data_list[idx]

        data_dict = self.get_data(data_path, ds)
        scene_name = data_dict["name"]

        data_dict = self.transform(data_dict)

        points_coord = data_dict["coord"]
        points_color = data_dict["color"]
        mask_proposal = data_dict["mask200"]

        bboxes = data_dict["bbox"]
        
        label = data_dict["segment"].long()
        unique_label = torch.unique(label).tolist()
        if -1 in unique_label:
            unique_label.remove(-1)
        if ds == "scannet20" and 19 in unique_label:
            unique_label.remove(19)
        if ds == "s3dis" and 12 in unique_label:
            unique_label.remove(12)

        inst_label = data_dict["instance"].long()
        unique_inst_label = torch.unique(inst_label).tolist()
        if -1 in unique_inst_label:
            unique_inst_label.remove(-1)

        classes_name, classes_dict = self.classes[ds]
        assert max(unique_label) < len(classes_name), f"{ds}, {max(unique_label)}, {len(classes_name)}"
        classes = [classes_name[class_id] for class_id in unique_label]

        sampled_class = random.choice(classes) # if vocab is None else vocab
        sem_id = classes_name.index(sampled_class)
        text = sampled_class

        distractor_classes = classes_dict[sampled_class]
        distractor_id = [classes_name.index(dc) for dc in distractor_classes]
            
        class_ids = [sem_id]
        sampled_ids = [sem_id]

        if sampled_class in classes:
            answer_template = random.choice(self.answer_list[1:])
        else:
            answer_template = self.answer_list[0]
        answers = [answer_template.format(class_name=text.lower())]

        question_templates = random.choice(self.question_list_sem)
        questions = [question_templates.format(class_name=text.lower())]
        assert len(questions) == len(answers) == len(class_ids)

        mask = (label == sem_id)
        unique_inst_ids = inst_label[mask].unique()
        if -1 in unique_inst_ids:
            unique_inst_ids.remove(-1)
        masks = [(inst_label == inst_id) for inst_id in unique_inst_ids]
        masks = torch.stack(masks, dim=0)

        target_boxes = [bboxes.get(inst_id, torch.zeros(8)) for inst_id in unique_inst_ids]

        distractor_mask = []
        distractor_boxes = []
        for id in distractor_id:
            d_mask = (label == id)
            distractor_unique_inst_ids = inst_label[d_mask].unique()
            distractor_mask += [(inst_label == inst_id) for inst_id in distractor_unique_inst_ids]
            distractor_boxes += [bboxes.get(inst_id, torch.zeros(8)) for inst_id in distractor_unique_inst_ids]        

        related_masks = distractor_mask
        target_boxes = target_boxes + distractor_boxes

        if len(related_masks) > 0:
            related_masks = torch.stack(related_masks, dim=0)

        if len(target_boxes) > 0:
            target_boxes = torch.stack(target_boxes, dim=0)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        if not "train" in self.split:
            conv.system = ""
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        return (
            scene_name,
            points_coord,
            points_color,
            None,
            mask_proposal,
            conversations,
            masks,
            related_masks,
            target_boxes,
            questions,
            None,
            sampled_ids,
        )

    def get_data(self, data_path, ds):
        name = self.get_data_name(data_path)

        valid_assets = self.VALID_ASSETS
        if ds in ["scannet20", "scannet200"]:
            class_num = ds[7:]
            valid_assets = [
                "segment{}".format(class_num) if item == "segment" else item 
                for item in self.VALID_ASSETS
                ]

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in valid_assets:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "mask200" in data_dict.keys():
            data_dict["mask200"] = data_dict["mask200"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        elif "segment" in data_dict.keys():
            if "scannetpp" in ds:
                data_dict["segment"] = data_dict["segment"][:, 0]
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if "instance" in data_dict.keys():
            if "scannetpp" in ds:
                data_dict["instance"] = data_dict["instance"][:, 0]
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        return data_dict

    def get_data_name(self, path):
        return os.path.basename(path)
