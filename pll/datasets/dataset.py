import os
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import sys
sys.path.append(ROOT_DIR)

from llms import conversation as conversation_lib
from utils import (DEFAULT_PC_END_TOKEN, DEFAULT_PC_START_TOKEN, 
                    DEFAULT_POINT_TOKEN, IGNORE_INDEX, POINT_TOKEN_INDEX)

from .sem_seg_dataset import SemSegDataset
from .refer_seg_dataset import ReferSegDataset
from .scene_cap_dataset import SceneCapDataset

def tokenizer_point_token(
        prompt, tokenizer, point_token_index=POINT_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<point>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [point_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    scene_path_list = []
    points_coord_list = []
    points_color_list = []
    points_index_list = []
    mask_proposal_list = []
    conversation_list = []
    masks_list = []
    related_masks_list = []
    target_boxes_list = []
    questions_list = []
    classes_list = []
    class_id_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        scene_name,
        points_coord,
        points_color,
        points_index,
        mask_proposal,
        conversations,
        masks,
        related_masks,
        target_boxes,
        questions,
        class_label,  # sampled class 
        class_id,  # class id
        inference,
    ) in batch:
        scene_path_list.append(scene_name)
        points_coord_list.append(points_coord.contiguous())
        points_color_list.append(points_color.contiguous())
        points_index_list.append(points_index)
        mask_proposal_list.append(mask_proposal)
        conversation_list.extend(conversations)
        classes_list.append(class_label)
        masks_list.append(masks.float())
        related_masks_list.append(related_masks)
        target_boxes_list.append(target_boxes)
        questions_list.append(questions)
        class_id_list.append(torch.tensor(class_id, dtype=torch.long))
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <point> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_POINT_TOKEN
            replace_token = (
                DEFAULT_PC_START_TOKEN + replace_token + DEFAULT_PC_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_POINT_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_point_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    # if conv_type == "llava_v1":
    #     sep = conv.sep + conv.roles[1] + ": "
    # else:
    #     sep = "[/INST] "
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_POINT_TOKEN in conversation:
                round_len = len(tokenizer_point_token(rou, tokenizer))
                instruction_len = len(tokenizer_point_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 150

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return_dict =  {
        "scene_paths": scene_path_list,
        "points_coord": points_coord_list,
        "points_color": points_color_list,
        "points_index": points_index_list,
        "mask_proposal": mask_proposal_list,
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "related_masks_list": related_masks_list,
        "target_boxes_list": target_boxes_list,
        "classes_list": classes_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "class_id_list": class_id_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

    return return_dict


class TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_pc_dir,
        tokenizer,
        transform=None,
        precision: str = "fp32",
        num_prompt_per_sample: int = 3,
        dataset="sem_seg",
        sample_rate=[1],
        sem_seg_data="scannet20",
        refer_seg_data="scanrefer",
        scene_cap_data="scenecap",
        refer_sample_rate="1",
        split="train",
    ):
        self.sample_rate = sample_rate
        self.base_pc_dir = base_pc_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split

        self.datasets = dataset.split("||")
        self.all_datasets = []
        self.global_indices = []  # Stores tuples (dataset_idx, local_idx)
        for i, ds_name in enumerate(self.datasets):
            if ds_name == "sem_seg":
                ds = SemSegDataset(
                    base_pc_dir,
                    tokenizer,
                    transform,
                    precision,
                    sem_seg_data,
                    split=self.split,
                )
            elif ds_name == "refer_seg":
                ds = ReferSegDataset(
                    base_pc_dir,
                    tokenizer,
                    transform,
                    precision,
                    refer_seg_data,
                    sampling_ratios=refer_sample_rate,
                    split=self.split,
                )
            elif ds_name == "scene_cap":
                ds = SceneCapDataset(
                    base_pc_dir,
                    tokenizer,
                    transform,
                    precision,
                    scene_cap_data,
                    split=self.split,
                )
            
            num_samples = int(len(ds) * self.sample_rate[i])
            sampled_indices = np.random.choice(len(ds), num_samples, replace=(self.sample_rate[i]>1))
            
            self.all_datasets.append(ds)
            self.global_indices.extend([(i, idx) for idx in sampled_indices])

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.global_indices[idx]
        ds = self.all_datasets[dataset_idx]
        inference = False if self.split == "train" else True
        return *ds[sample_idx], inference
    

class ValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_pc_dir,
        tokenizer,
        transform=None,
        precision: str = "fp32",
        num_prompt_per_sample: int = 3,
        dataset="refer_seg||scanrefer",
        split="train",
        vocab_list=None,
    ):
        self.base_pc_dir = base_pc_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.split = split

        self.datasets = dataset.split("||")
        assert len(self.datasets) == 2
        ds_name = self.datasets[0]
        if ds_name == "sem_seg":
            self.ds = SemSegDataset(
                base_pc_dir,
                tokenizer,
                transform,
                precision,
                sem_seg_data=self.datasets[1],
                split=self.split,
                vocab_list=vocab_list,
            )
        elif ds_name == "refer_seg":
            self.ds = ReferSegDataset(
                base_pc_dir,
                tokenizer,
                transform,
                precision,
                refer_seg_data=self.datasets[1],
                split=self.split,
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ds = self.ds
        inference = False if self.split == "train" else True
        return *ds[idx], inference