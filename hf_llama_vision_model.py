import os
import torch
from PIL import Image
from transformers import (
    LlamaForCausalLM, 
    AutoTokenizer, 
    CLIPVisionModel, 
    CLIPImageProcessor, 
    PretrainedConfig, 
    PreTrainedModel
)

class LlamaVisionConfig(PretrainedConfig):
    model_type = "llama_vision"
    def __init__(
        self, 
        llama_name="meta-llama/Llama-3.2-1B", 
        clip_name="openai/clip-vit-large-patch14", 
        **kwargs
    ):
        self.llama_name = llama_name
        self.clip_name = clip_name
        super().__init__(**kwargs)

class LlamaVisionModel(PreTrainedModel):
    config_class = LlamaVisionConfig

    def __init__(self, config):
        super().__init__(config)
        # Load the base models inside the class
        self.llama = LlamaForCausalLM.from_pretrained(
            config.llama_name, 
            dtype=torch.bfloat16
        )
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            config.clip_name, 
            dtype=torch.bfloat16
        )
        
        # Projector: Maps CLIP hidden size to Llama hidden size
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.vision_encoder.config.hidden_size, self.llama.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.llama.config.hidden_size, self.llama.config.hidden_size)
        ).to(torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
    
        img_feats = self.vision_encoder(pixel_values).last_hidden_state
        img_embeds = self.projector(img_feats)
        
        
        text_embeds = self.llama.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=input_ids.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1) if attention_mask is not None else None


        if labels is not None:
            ignore_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=input_ids.device)
            full_labels = torch.cat([ignore_labels, labels], dim=1)
        else:
            full_labels = None

        return self.llama(
            inputs_embeds=inputs_embeds, 
            attention_mask=full_mask, 
            labels=full_labels
        )
    def genrate_with_vision(self,img_path, query,tokenizer,image_processor,device="cuda",max_new_tokens=128,temperature=0.8,top_p=0.9,repetition_penalty=1.1,):
        raw_image = Image.open(img_path).convert("RGB")
        

        pixel_values = image_processor(raw_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

        prompt = query#f"{query}\nAssistant: <image>"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            img_feats = self.vision_encoder(pixel_values).last_hidden_state
            img_embeds = self.projector(img_feats)
            text_embeds = self.llama.get_input_embeddings()(inputs.input_ids)

            inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)

            output_ids = self.llama.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()

