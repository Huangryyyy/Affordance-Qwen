import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from utils import *
import os
import torch
import json


class AGD20KwithDepth(Dataset):
    def __init__(
        self,
        json_dir,
        data_dir="/mnt/DATA/AGD20K",
        scale_size=896,
        load_depth_image=True,
    ):
        super(AGD20KwithDepth, self).__init__()
        self.json_dir = json_dir
        with open(self.json_dir, "r", encoding="utf-8") as f:
            self.json_data = json.load(f)
        self.data_dir = data_dir
        self.scale_size = scale_size
        self.load_depth_image = load_depth_image

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, i):
        sample = self.json_data[i].copy()
        img = Image.open(os.path.join(self.data_dir, sample["image_path"])).convert(
            "RGB"
        )
        img = expand2square(img)
        img = img.resize((self.scale_size, self.scale_size), resample=Image.BICUBIC)
        mask = Image.open(os.path.join(self.data_dir, sample["gt_path"])).convert("L")
        mask = expand2square(mask)
        mask = mask.resize((self.scale_size, self.scale_size), resample=Image.BICUBIC)
        mask = np.array(mask)
        if mask.max() > 1:
            gt_mask = (mask > 127).astype(np.float32)
        else:
            gt_mask = mask.astype(np.float32)
        if self.load_depth_image:
            depth_img = Image.open(
                os.path.join(self.data_dir, sample["depth"])
            ).convert("RGB")
            depth_img = expand2square(depth_img)
            depth_img = depth_img.resize(
                (self.scale_size, self.scale_size), resample=Image.BICUBIC
            )
        else:
            depth_img = None
        inputs = {
            "id": sample["id"],
            "image": img,
            "depth_image": depth_img,
            "query": sample["query"],
            "answer": sample["answer"].replace("<|extra_0|>", "<seg_token>"),
            "gt_mask": gt_mask,
        }
        return inputs


class CollatorForQwen2_5:
    def __init__(self, processor, ignore_index=-100):
        self.processor = processor
        self.ignore_index = ignore_index

    def add_labels(self, inputs):
        inputs["labels"] = []
        for i in range(len(inputs["input_ids"])):
            input_ids = inputs["input_ids"][i]
            labels = input_ids.clone()
            assistant_role_id = self.processor.tokenizer.convert_tokens_to_ids(
                "assistant"
            )
            sep_indices = (input_ids == assistant_role_id).nonzero(as_tuple=True)[0]
            if len(sep_indices) > 0:
                start_index = sep_indices[-1].item()
                labels[: start_index + 2] = (
                    self.ignore_index
                )  # the format is "assistant\n{answer}", +2 for the '\n'
            if "attention_mask" in inputs:
                padding_mask = inputs["attention_mask"][i] == 0
                labels[padding_mask] = self.ignore_index
            inputs["labels"].append(labels)
        inputs["labels"] = torch.stack(inputs["labels"])

    def __call__(self, features):
        batch_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": features[i]["image"]},
                        {"type": "text", "text": features[i]["query"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": features[i]["answer"]}],
                },
            ]
            for i in range(len(features))
        ]
        batch_inputs = self.processor.apply_chat_template(
            batch_messages,
            padding=True,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # btw, because of the speical design of position_ids in Qwen2_5, we don't manually input the 1d position_ids

        if features[0]["depth_image"] is not None:
            batch_depth_messages = [
                [  # just create messages for get the pixel_values and "image_grid_thw" for depth images
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": features[i]["depth_image"]},
                            {"type": "text", "text": " "},
                        ],
                    }
                ]
                for i in range(len(features))
            ]
            batch_depth_inputs = self.processor.apply_chat_template(
                batch_depth_messages,
                padding=True,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            batch_inputs["depth_pixel_values"] = batch_depth_inputs["pixel_values"]
            batch_inputs["depth_image_grid_thw"] = batch_depth_inputs["image_grid_thw"]
        batch_inputs["gt_masks"] = torch.stack(
            [torch.Tensor(features[i]["gt_mask"]) for i in range(len(features))]
        )
        self.add_labels(batch_inputs)
        batch_inputs["sample_ids"] = [features[i]["id"] for i in range(len(features))]
        return batch_inputs
