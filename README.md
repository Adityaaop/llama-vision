# LlamaVision-llama-3.2-1b


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
import torch
from transformers import AutoTokenizer, CLIPImageProcessor,AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "iamthehimansh/LlamaVision-llama-3.3-1b"
external_image_path = r"./image/bahubali_poster.jpg" 

model = AutoModel.from_pretrained(
    "iamthehimansh/LlamaVision-llama-3.3-1b", 
    trust_remote_code=True
)
# LlamaVisionModel.from_pretrained(model_path)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
image_processor = CLIPImageProcessor.from_pretrained(model_path)

query="Explain this image "
res= model.genrate_with_vision(external_image_path, query,tokenizer,image_processor)
print(res)
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