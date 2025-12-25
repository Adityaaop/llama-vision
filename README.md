# LlamaVision-llama-3.2-1b

---

language: en
license: llama3.2
tags:

* multimodal
* image-to-text
* vision-language
* llama
* clip
datasets:
* jackyhate/text-to-image-2M
pipeline_tag: image-to-text
library_name: transformers

---

## Model Overview

**LlamaVision** is a custom multimodal vision-language model that bridges visual understanding with advanced text generation. It utilizes a **CLIP-ViT-Large-Patch14** vision encoder and a **Llama-3.2-1B** language backbone, connected via a trainable linear projector. The model is designed to describe images in detail and answer questions about visual content based on provided prompts.

## Architecture

The model follows a modular architecture:

* **Vision Encoder:** [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14) (OpenAI), which encodes images into 1024-dimensional visual features.
* **Language Backbone:** [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) (Meta), an auto-regressive transformer optimized for efficient inference and multilingual dialogue.
* **Multimodal Projector:** A custom linear projection layer that maps visual embeddings into the Llama-3.2-1B hidden space (2048 dimensions).

## Training Details

* **Dataset:** Fine-tuned on a subset of the [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset, specifically utilizing high-resolution image-prompt pairs for alignment.
* **Objective:** Causal Language Modeling (CLM) loss, where the model learns to predict descriptive text conditioned on visual embeddings.
* **Precision:** Trained using `bfloat16` for memory efficiency and numerical stability.

## Intended Use

* **Direct Use:** Automated image captioning and detailed scene description.
* **Downstream Use:** Visual Question Answering (VQA) and multimodal assistants.
* **Out-of-Scope:** Not intended for high-stakes medical diagnosis, surveillance, or any activity prohibited by the [Llama 3.2 Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy).

## How to Get Started

You can load and use the model with the following Python code :

```python
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from hf_model import LlamaVisionModel
# git clone github.com/iamthehimansh/LLAMA-VISION to import LlamaVisionModel from hf_model file 
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "iamthehimansh/LlamaVision-llama-3.3-1b"
external_image_path = r"./image/bahubali_poster.jpg" 

model = LlamaVisionModel.from_pretrained(model_path)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
image_processor = CLIPImageProcessor.from_pretrained(model_path)

def chat_with_vision(img_path, query):
    raw_image = Image.open(img_path).convert("RGB")
    pixel_values = image_processor(raw_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

    prompt = query#f"{query}\nAssistant: <image>"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        img_feats = model.vision_encoder(pixel_values).last_hidden_state
        img_embeds = model.projector(img_feats)
        text_embeds = model.llama.get_input_embeddings()(inputs.input_ids)

        inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)

        output_ids = model.llama.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    if os.path.exists(external_image_path):
        while True:
            user_q = input("\nQuestion about image: ")
            if user_q.lower() == 'exit':
                break
            
            try:
                answer = chat_with_vision(external_image_path, user_q)
                print(f"\nAI: {answer}")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        print(f"Error: Image not found at {external_image_path}")
```

## Citation and Acknowledgments

If you use this model, please cite the base components:

* **Llama 3.2:** Meta Platforms, Inc.
* **CLIP:** OpenAI
* **Dataset:** `jackyhate/text-to-image-2M`


## MODEL STATISTICS
- Total Parameters:      1,545,289,728
- Trainable Parameters:  6,295,552
- Non-Trainable:         1,538,994,176