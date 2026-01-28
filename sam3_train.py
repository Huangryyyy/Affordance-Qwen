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

args_path = "./args/sam3_train.yaml"
with open(args_path, "r") as f:
    args_dict = yaml.safe_load(f)
    print(args_dict)
args = Config(args_dict)

os.makedirs(args.save_model_path, exist_ok=True)
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
    json_dir=args.train_json_path,
    data_dir=args.img_path,
    scale_size=args.image_size,
    load_depth_image=False,
    additional_process=additional_process,
)
data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=CollatorForSam3(processor=processor),
)

model.load_state_dict(torch.load(os.path.join(args.load_model_path, "affordance_sam3.pt")))

for name, param in model.named_parameters():
    if "vision_encoder" in name:
        param.requires_grad = False
    if param.requires_grad:
        print(name)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
optimizer.zero_grad()
global_step = 0
for epoch in range(args.epochs):
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
        gt_masks = inputs["gt_masks"].to(pred_masks_upscaled.dtype)

        mask_loss, score_loss, batch_best_pred = compute_bestpred_loss(
            pred_masks_upscaled, outputs.pred_logits, gt_masks
        )
        loss=mask_loss+score_loss
        loss.backward()
        if global_step % 10 == 0:
            print(f"Epoch {epoch} Step {global_step}:")
            print(f"Loss: {loss.item()}")
            print(f"sample ids: {inputs['sample_ids'][0]}")
            action, thing, file_name = split_id(inputs["sample_ids"][0])
            save_file_path = os.path.join(
                args.output_image_path,
                action,
                thing,
                f"step{global_step}_{file_name}",
            )
            save_example(
                batch_best_pred[0],
                inputs["gt_masks"][0],
                inputs["origin_images"][0],
                file_path=save_file_path,
            )

        global_step += 1

        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if global_step % 2000 == 0:
            torch.save(
                model.state_dict(),
                f"{args.save_model_path}/affordance_sam3_step{global_step}.pt",
            )


torch.save(model.state_dict(), f"{args.save_model_path}/affordance_sam3.pt")
processor.save_pretrained(args.save_model_path)
