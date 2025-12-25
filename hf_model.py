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

from hf_llama_vision_model import LlamaVisionConfig,LlamaVisionModel



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