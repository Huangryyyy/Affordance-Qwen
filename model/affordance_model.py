import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from model.mask_decoder import MaskDecoder
from model.prompt_encoder import PromptEncoder
from model.twowaytrans import TwoWayTransformer
from model.common import LayerNorm2d


class AffordanceModelSam(nn.Module):
    def __init__(self, image_size, image_feature_dim, text_feature_dim, patch_size):
        super().__init__()
        self.image_feature_dim = image_feature_dim
        self.text_feature_dim = text_feature_dim
        self.transformer_dim = 256
        self.image_size = image_size
        self.vit_patch_size = patch_size
        self.image_embedding_size = self.image_size // self.vit_patch_size

        self.image_neck = nn.Sequential(
            nn.Conv2d(
                in_channels=self.image_feature_dim,
                out_channels=self.image_feature_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            LayerNorm2d(self.image_feature_dim),
            nn.Conv2d(
                in_channels=image_feature_dim,
                out_channels=self.transformer_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.transformer_dim),
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.transformer_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=self.transformer_dim,
        )

        self.prompt_encoder = PromptEncoder(
            embed_dim=self.transformer_dim,
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )

        self.text_neck = nn.Sequential(
            nn.Linear(self.text_feature_dim, self.text_feature_dim),
            nn.ReLU(),
            nn.Linear(self.text_feature_dim, self.transformer_dim),
        )

    def forward(self, image_features, text_features):
        image_features = self.image_neck(image_features)
        text_features = self.text_neck(text_features)
        (sparse_embeddings, dense_embeddings) = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=text_features.unsqueeze(1),
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks, iou_predictions


class AffordanceQwen2_5(nn.Module):
    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        seg_token_id,
        image_size,
        use_depth_image=False,
    ):
        super().__init__()
        self.base_model = base_model
        self.seg_token_id = seg_token_id
        self.image_size = image_size
        patch_size = 28  # actually the real patch_size in Qwen2_5 is 14, but the number of tokens corresponds to patch_size=28
        self.patch_shape = image_size // patch_size
        self.image_feature_dim = 3584 * (2 if use_depth_image else 1)
        self.affordance_decoder = AffordanceModelSam(
            image_size=image_size,
            image_feature_dim=self.image_feature_dim,
            text_feature_dim=3584,
            patch_size=patch_size,
        )
        self.use_depth_image = use_depth_image

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values,
        image_grid_thw,
        depth_pixel_values=None,
        depth_image_grid_thw=None,
        **kwargs,
    ):
        image_features = self.base_model.visual(pixel_values, grid_thw=image_grid_thw)
        kwargs.pop("output_hidden_states", None)
        if (
            self.use_depth_image
            and depth_pixel_values is not None
            and depth_image_grid_thw is not None
        ):
            depth_image_features = self.base_model.visual(
                depth_pixel_values, grid_thw=depth_image_grid_thw
            )
            image_features = torch.cat([image_features, depth_image_features], dim=-1)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **kwargs,
        )
        image_features = image_features.view(
            -1, self.patch_shape, self.patch_shape, self.image_feature_dim
        ).permute(0, 3, 1, 2)
        seg_token_mask = input_ids == self.seg_token_id
        text_features = outputs.hidden_states[-1][seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)
        image_features = torch.repeat_interleave(
            image_features, seg_token_counts, dim=0
        )  # deal with the situation that one sample includes several seg_tokens
        low_res_masks, iou_predictions = self.affordance_decoder(
            image_features, text_features
        )
        low_res_masks = low_res_masks[:, 0]
        return {
            "sft_loss": outputs.loss,
            "base_model_outputs": outputs,
            "pred_masks": low_res_masks,
        }
