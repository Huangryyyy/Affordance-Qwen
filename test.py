import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration,
)
import yaml
import os
from utils import *
from dataset import AGD20KwithDepth, CollatorForQwen2_5
from model.affordance_model import AffordanceQwen2_5
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

set_seed(seed=42)

args_path = "./args/test.yaml"
with open(args_path, "r") as f:
    args_dict = yaml.safe_load(f)
    print(args_dict)
args = Config(args_dict)

os.makedirs(args.output_image_path, exist_ok=True)

processor = AutoProcessor.from_pretrained(args.load_model_path)
seg_token_id = processor.tokenizer(
    "<seg_token>", return_tensors="pt", add_special_tokens=False
)["input_ids"][0].item()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, device_map="auto", dtype=torch.bfloat16
)
model.resize_token_embeddings(len(processor.tokenizer))
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
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)

train_dataset = AGD20KwithDepth(
    json_dir=args.test_json_path,
    data_dir=args.img_path,
    scale_size=args.image_size,
    load_depth_image=False,
)
data_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=CollatorForQwen2_5(processor),
)

affordance_model = AffordanceQwen2_5(
    model, seg_token_id=seg_token_id, image_size=args.image_size
)
affordance_model.load_state_dict(
    torch.load(os.path.join(args.load_model_path, "affordance_qwen.pt"))
)
kl_arr, sim_arr, nss_arr = [], [], []
for step, inputs in enumerate(data_loader):
    inputs = {
        k: (
            v.to(affordance_model.base_model.device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in inputs.items()
    }
    outputs = affordance_model(**inputs)
    pred_masks_upscaled = (
        F.interpolate(
            outputs["pred_masks"].unsqueeze(1),
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(1)
        .sigmoid()
    )
    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        decoded_input = processor.tokenizer.decode(
            input_ids[0][-20:], skip_special_tokens=False
        )
        print(f"Input Example:\n{decoded_input.replace("\n","\\n")}")
    logits = outputs["logits"]
    pred_tokens = torch.argmax(logits[0], dim=-1)
    decoded_pred = processor.tokenizer.decode(
        pred_tokens[-20:], skip_special_tokens=False
    )
    print(f"Output Example:\n{decoded_pred.replace("\n","\\n")}")
    kl_arr.append(cal_kl(pred_masks_upscaled, inputs["gt_masks"]).item())
    sim_arr.append(cal_sim(pred_masks_upscaled, inputs["gt_masks"]).item())
    nss_arr.append(cal_nss(pred_masks_upscaled, inputs["gt_masks"]).item())
    print(f"KL: {kl_arr[-1]}   SIM: {sim_arr[-1]}   NSS: {nss_arr[-1]}")
    action, thing, file_name = split_id(inputs["sample_ids"][0])
    save_file_path = os.path.join(
        args.output_image_path,
        action,
        thing,
        file_name,
    )
    save_example(
        pred_masks_upscaled[0],
        inputs["gt_masks"][0],
        inputs["origin_images"][0],
        file_path=save_file_path,
    )
print(
    f"FINAL RESULT:   KL: {np.mean(kl_arr)}   SIM: {np.mean(sim_arr)}   NSS: {np.mean(nss_arr)}"
)
