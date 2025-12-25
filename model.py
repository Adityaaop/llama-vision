import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import LlamaForCausalLM, CLIPVisionModel

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, image_processor, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.filenames = list(set([f.split('.')[0] for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]
        img_path = os.path.join(self.data_dir, f"{base_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

        json_path = os.path.join(self.data_dir, f"{base_name}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        description = data['prompt']

        tokens = self.tokenizer(
            description, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0)
        }

class LlamaVisionModel(torch.nn.Module):
    def __init__(self, llama_name, clip_name):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(llama_name, dtype=torch.bfloat16)
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_name, dtype=torch.bfloat16)
        
        # Freeze LLM and Vision Encoder
        for p in self.llama.parameters(): p.requires_grad = False
        for p in self.vision_encoder.parameters(): p.requires_grad = False

        # Projector: Maps CLIP (1024) to Llama (2048/4096)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.vision_encoder.config.hidden_size, self.llama.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.llama.config.hidden_size, self.llama.config.hidden_size)
        ).to(torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feats = self.vision_encoder(pixel_values).last_hidden_state
        img_embeds = self.projector(img_feats)
        text_embeds = self.llama.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=input_ids.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)

        ignore_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=input_ids.device)
        full_labels = torch.cat([ignore_labels, input_ids], dim=1)

        outputs = self.llama(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)
        return outputs.loss