# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


import numpy as np
import torch
import torch.utils.checkpoint
from transformers.image_processing_utils import select_best_resolution
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)

from QEfficient.utils._utils import IOInfo
from QEfficient.utils.logging_utils import logger

BS = 1
NUM_CHANNEL = 3
image_num_patches = 10
SEQ_LEN = 5500
CTX_LEN = 6000


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    # breakpoint()
    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    # breakpoint()
    # height=1152
    # width=1152
    # return 3, 3
    return height // patch_size, width // patch_size


# def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
#     if not isinstance(grid_pinpoints, list):
#         raise TypeError("grid_pinpoints should be a list of tuples or lists")

#     # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
#     if not isinstance(image_size, (list, tuple)):
#         if not isinstance(image_size, (torch.Tensor, np.ndarray)):
#             raise TypeError(f"image_size invalid type {type(image_size)} with value {image_size}")
#         image_size = image_size.tolist()


#     best_resolution = select_best_resolution(image_size, grid_pinpoints)
#     height, width = best_resolution
#     num_patches = 0
#     # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
#     for i in range(0, height, patch_size):
#         for j in range(0, width, patch_size):
#             num_patches += 1
#     # add the base patch
#     num_patches += 1
#     return num_patches
def unpad_image(tensor, original_size):
    # breakpoint()
    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid, should be either list, tuple, np.ndarray or tensor"
            )
        original_size = original_size.tolist()
    # breakpoint()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]
    if torch.is_tensor(current_height):
        current_height = current_height.item()
        current_width = current_width.item()
    # current_height=81
    # current_width=81

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    # original_aspect_ratio=1.4517583408476105
    # current_aspect_ratio=1.0
    # breakpoint()
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        # new_height=55
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        # breakpoint()
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class QEffLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    # def pack_image_features(self,image_features,image_sizes,vision_feature_select_strategy, image_newline=None):
    #     # breakpoint()
    #     new_image_features = []
    #     feature_lens = []
    #     for image_idx, image_feature in enumerate(image_features):
    #         if image_feature.shape[0] > 1:
    #             base_image_feature = image_feature[0]
    #             image_feature = image_feature[1:]
    #             height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

    #             num_patch_height, num_patch_width = get_anyres_image_grid_shape(
    #                 image_sizes[image_idx],
    #                 self.config.image_grid_pinpoints,
    #                 self.config.vision_config.image_size,
    #             )
    #             # breakpoint()
    #             if (
    #                 np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
    #                 and vision_feature_select_strategy == "default"
    #             ):
    #                 logger.warning_once(
    #                     "Image feature shape does not line up with the provided patch size. "
    #                     "You may be using the `default` vision_feature_select_strategy with a"
    #                     " visual encoder that does not have CLS."
    #                 )

    #             image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #             image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #             image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #             image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #             if self.image_newline is not None:
    #                 image_feature = torch.cat(
    #                     (
    #                         image_feature,
    #                         self.image_newline[:, None, None]
    #                         .expand(*image_feature.shape[:-1], 1)
    #                         .to(image_feature.device, image_feature.dtype),
    #                     ),
    #                     dim=-1,
    #                 )
    #             image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #             image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #         else:
    #             image_feature = image_feature[0]
    #             if self.image_newline is not None:
    #                 image_feature = torch.cat((image_feature, self.image_newline[None].to(image_feature)), dim=0)
    #         new_image_features.append(image_feature)
    #         feature_lens.append(image_feature.size(0))
    #     image_features = torch.cat(new_image_features, dim=0)
    #     feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
    #     return image_features, feature_lens

    def forward(self, input_ids, image_sizes, position_ids, pixel_values, past_key_values):
        # breakpoint()
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # breakpoint()
        # image_num_patches = [
        #     image_size_to_num_patches(
        #         image_size=imsize,
        #         grid_pinpoints=self.config.image_grid_pinpoints,
        #         patch_size=self.config.vision_config.image_size,
        #     )
        #     for imsize in image_sizes
        # ]
        for imsize in image_sizes:
            if not isinstance(self.config.image_grid_pinpoints, list):
                raise TypeError("grid_pinpoints should be a list of tuples or lists")
            if not isinstance(imsize, (list, tuple)):
                if not isinstance(imsize, (torch.Tensor, np.ndarray)):
                    raise TypeError(f"image_size invalid type {type(imsize)} with value {imsize}")
                imsize = imsize.tolist()
            best_resolution = select_best_resolution(imsize, self.config.image_grid_pinpoints)
            height, width = best_resolution
            num_patches = 0
            for i in range(0, height, self.config.vision_config.image_size):
                for j in range(0, width, self.config.vision_config.image_size):
                    num_patches += 1
            # add the base patch
            num_patches += 1
        image_num_patches = [num_patches]
        # breakpoint()

        if pixel_values.dim() == 5:
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values_new = torch.cat(_pixel_values_list, dim=0)

        # Image features
        image_feature = self.vision_tower(pixel_values_new, output_hidden_states=True)
        # breakpoint()
        if isinstance(self.config.vision_feature_layer, int):
            selected_image_feature = image_feature.hidden_states[self.config.vision_feature_layer]
        else:
            hs_pool = [image_feature.hidden_states[layer_idx] for layer_idx in self.config.vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        vision_feature_select_strategy = self.config.vision_feature_select_strategy
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)

        image_features = torch.split(image_features, image_num_patches, dim=0)
        # image_features, _ = self.pack_image_features(
        #         image_features,
        #         image_sizes=image_sizes,
        #         vision_feature_select_strategy=vision_feature_select_strategy,
        #         image_newline=self.image_newline,
        #     )
        new_image_features = []
        # feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                # breakpoint()
                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = torch.cat((image_feature, self.image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
        image_features = torch.cat(new_image_features, dim=0)
        # breakpoint()
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        mask = input_ids == self.config.image_token_index
        indices1 = mask.to(torch.int64).cumsum(1) - 1
        # indices0 = torch.arange(mask.shape[0]).view(-1, 1)
        image_features_expanded = image_features[indices1]
        image_inputs_embeds = torch.where(mask.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # *where to skip image encoder for decode*
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_inputs_embeds)
        # breakpoint()
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        # breakpoint()
        return outputs.logits, pixel_values, outputs.past_key_values

    def get_dummy_inputs(self, **kwargs):
        # breakpoint()
        num_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.hidden_size // self.config.text_config.num_attention_heads
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 384)
        else:
            img_size = 384
        breakpoint()
        inputs = {
            "input_ids": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
            "attention_mask": torch.ones((BS, SEQ_LEN), dtype=torch.int64),
            "pixel_values": torch.zeros((BS, image_num_patches, NUM_CHANNEL, img_size, img_size), dtype=torch.float32),
            "image_sizes": torch.ones((1, 2), dtype=torch.float32),
        }
        inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1)
        inputs["past_key_values"] = []
        for i in range(num_layers):
            inputs["past_key_values"].append(
                (
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                    torch.zeros(BS, num_key_value_heads, CTX_LEN, head_dim),
                )
            )
        inputs["position_ids"] = torch.full(inputs["position_ids"].shape, CTX_LEN - 1)
        return inputs

    def get_specializations(
        self, batch_size: int, prefill_seq_len: int, ctx_len: int, img_size: int, **compiler_options
    ):
        max_num_images = compiler_options.pop("max_num_images", 1)
        prefill_seq_len = prefill_seq_len if prefill_seq_len else SEQ_LEN
        ctx_len = ctx_len if ctx_len else CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 384
            logger.warning("Setting img_size to be 336, as it was neither passed nor found in vision_config")

        specializations = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
            },
            {
                "batch_size": batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "max_num_images": max_num_images,
                "img_size": img_size,
            },
        ]
        return specializations, compiler_options

    def get_onnx_dynamic_axes(
        self,
    ):
        # Define dynamic axes
        num_layers = self.config.text_config.num_hidden_layers

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "batch_size", 3: "img_size", 4: "img_size"},
        }
        for i in range(num_layers):
            dynamic_axes[f"past_key.{i}"] = {0: "batch_size", 2: "ctx_len"}
            dynamic_axes[f"past_value.{i}"] = {0: "batch_size", 2: "ctx_len"}

        return dynamic_axes

    def get_output_names(
        self,
    ):
        output_names = ["logits", "pixel_values_RetainedState"]
        for i in range(self.language_model.config.num_hidden_layers):
            for kv in ["key", "value"]:
                output_names.append(f"past_{kv}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("batch_size", 3, "img_size", "img_size")),
        ]
