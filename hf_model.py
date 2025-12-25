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
            torch_dtype=torch.bfloat16
        )
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            config.clip_name, 
            torch_dtype=torch.bfloat16
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



def export_full_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_directory = "./LlamaVision_model"
    
    
    llama_name = "meta-llama/Llama-3.2-1B"
    clip_name = "openai/clip-vit-large-patch14"
    projector_weights_path = "llama_vision_projector.pt"

    print("Initializing full model...")
    config = LlamaVisionConfig(llama_name=llama_name, clip_name=clip_name)
    full_model = LlamaVisionModel(config)

    if os.path.exists(projector_weights_path):
        print(f"Loading projector weights from {projector_weights_path}...")
        state_dict = torch.load(projector_weights_path, map_location=device)
        full_model.projector.load_state_dict(state_dict)
    else:
        print(f"Warning: {projector_weights_path} not found. Exporting base projector weights.")

    print("Saving tokenizer and processors...")
    tokenizer = AutoTokenizer.from_pretrained(llama_name)
    image_processor = CLIPImageProcessor.from_pretrained(clip_name)
    
    tokenizer.save_pretrained(save_directory)
    image_processor.save_pretrained(save_directory)

    print(f"Saving full model to {save_directory}...")
    full_model.save_pretrained(save_directory)
    
    print("\nExport Complete!")
# 1. Register the classes for the Hub
LlamaVisionConfig.register_for_auto_class()
LlamaVisionModel.register_for_auto_class("AutoModel")

def push_to_hf_hub(repo_id):
    # Setup paths
    llama_name = "meta-llama/Llama-3.2-1B"
    clip_name = "openai/clip-vit-large-patch14"
    projector_weights = "llama_vision_projector.pt"
    
    # Initialize model
    config = LlamaVisionConfig(llama_name=llama_name, clip_name=clip_name)
    model = LlamaVisionModel(config)
    
    # Load your trained projector weights
    if os.path.exists(projector_weights):
        model.projector.load_state_dict(torch.load(projector_weights))
    
    # Push to Hub
    # This will upload the weights AND the Python code files
    model.push_to_hub(repo_id, private=False)
    
    # Push tokenizer and processor too
    tokenizer = AutoTokenizer.from_pretrained(llama_name)
    image_processor = CLIPImageProcessor.from_pretrained(clip_name)
    
    tokenizer.push_to_hub(repo_id)
    image_processor.push_to_hub(repo_id)
    

# Usage
# push_to_hf_hub("my-llama-vision-model")

if __name__ == "__main__":
    # export_full_model()
    push_to_hf_hub("iamthehimansh/LlamaVision-llama-3.3-1b")