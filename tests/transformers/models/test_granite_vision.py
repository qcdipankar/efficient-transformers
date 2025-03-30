# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    TextStreamer,
)

from QEfficient import QEFFAutoModelForImageTextToText  # noqa: E402

model_id = "ibm-granite/granite-vision-3.2-2b"

config = AutoConfig.from_pretrained(model_id)

# config.text_config.num_hidden_layers = 1
# config.vision_config.num_hidden_layers = 10
py_model = AutoModelForVision2Seq.from_pretrained(model_id, low_cpu_mem_usage=True, config=config)
breakpoint()
processor = AutoProcessor.from_pretrained(model_id)
img_path = hf_hub_download(repo_id=model_id, filename="example.png")
# img_path="http://images.cocodataset.org/val2017/000000039769.jpg"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "Describe the image"},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
)

# Pytorch output

# output = py_model.generate(**inputs, max_new_tokens=128, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))
# print(output)


# img_path="/home/dipankar/test_qeff/efficient-transformers/tests/transformers/models/test1.jpg"
# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image")

model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, kv_offload=False, config=config)
model.compile(num_devices=4, img_size=384, prefill_seq_len=1024, ctx_len=6000)
streamer = TextStreamer(processor.tokenizer)
output = model.generate(inputs=inputs, device_ids=[0, 1, 2, 3], generation_len=128, runtime_ai100=True)

# QEff pytorch

# print(output)
# print(processor.tokenizer.batch_decode(output))
# print(output)
# breakpoint()

# AIC Output

print(output.generated_ids)
print(processor.tokenizer.batch_decode(output.generated_ids))
print(output)
