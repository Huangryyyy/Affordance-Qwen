import torch
from transformers import Sam3Processor, Sam3Model
import yaml
import os
from utils import *
from dataset import AGD20KwithDepth, CollatorForSam3
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import re
set_seed(seed=42)

args_path = "./args/sam3_test.yaml"
with open(args_path, "r") as f:
    args_dict = yaml.safe_load(f)
    print(args_dict)
args = Config(args_dict)

os.makedirs(args.output_image_path, exist_ok=True)

from transformers import logging

logging.disable_progress_bar()

model = Sam3Model.from_pretrained(args.model_path, dtype=torch.bfloat16).to("cuda")
processor = Sam3Processor.from_pretrained(args.model_path)

if args.use_lora:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

def additional_process(sample):
    pattern = r"What\s+(.*?)\s+should we interact with in order to\s+(.*?)\s+it\?"
    sample["query"]=re.sub(pattern, r"\1 to \2", sample["query"])
    return sample
train_dataset = AGD20KwithDepth(
    json_dir=args.test_json_path,
    data_dir=args.img_path,
    scale_size=args.image_size,
    load_depth_image=False,
    additional_process=additional_process,
)
data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=CollatorForSam3(processor=processor),
)
model.load_state_dict(torch.load(os.path.join(args.load_model_path, "affordance_sam3.pt")))

kl_arr, sim_arr, nss_arr = [], [], []
for step, inputs in enumerate(data_loader):
    inputs = {
        k: (
            v.to(model.device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in inputs.items()
    }
    outputs = model(**inputs)
    pred_masks_upscaled = (
        F.interpolate(  # scale to the image size
            outputs.pred_masks,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
    )
    max_idx=torch.argmax(outputs.pred_logits, dim=1)
    batch_best_pred = pred_masks_upscaled[torch.arange(pred_masks_upscaled.shape[0]), max_idx].sigmoid()
    kl_arr.extend(cal_kl(batch_best_pred, inputs["gt_masks"]).cpu().detach().numpy().tolist())
    sim_arr.extend(cal_sim(batch_best_pred, inputs["gt_masks"]).cpu().detach().numpy().tolist())
    nss_arr.extend(cal_nss(batch_best_pred, inputs["gt_masks"]).cpu().detach().numpy().tolist())
    print(f"KL: {kl_arr[-1]}   SIM: {sim_arr[-1]}   NSS: {nss_arr[-1]}")
    action, thing, file_name = split_id(inputs["sample_ids"][0])
    save_file_path = os.path.join(
        args.output_image_path,
        action,
        thing,
        file_name,
    )
    save_example(
        batch_best_pred[0],
        inputs["gt_masks"][0],
        inputs["origin_images"][0],
        file_path=save_file_path,
    )
print(
    f"FINAL RESULT:   KL: {np.mean(kl_arr)}   SIM: {np.mean(sim_arr)}   NSS: {np.mean(nss_arr)}"
)