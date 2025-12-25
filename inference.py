import os
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from model import LlamaVisionModel

# Config
llama_name = "meta-llama/Llama-3.2-1B"
clip_name = "openai/clip-vit-large-patch14"
device = "cuda"
external_image_path = r".//bahubali_poster.jpg"

# Load Models
tokenizer = AutoTokenizer.from_pretrained(llama_name)
image_processor = CLIPImageProcessor.from_pretrained(clip_name)
model = LlamaVisionModel(llama_name, clip_name).to(device)

# Load Trained Weights
if os.path.exists("llama_vision_projector.pt"):
    model.projector.load_state_dict(torch.load("llama_vision_projector.pt", map_location=device))
    print("Projector weights loaded.")
model.eval()

def chat_with_image(image_path, user_query):
    raw_image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(raw_image, return_tensors="pt").pixel_values.to(device, dtype=torch.bfloat16)

    # We place the query before the image to give context
    prompt = f"{user_query}"#"\n <image>"
    prefix_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    asst=""#"</image> "
    suffix_ids=tokenizer(asst, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        img_feats = model.vision_encoder(pixel_values).last_hidden_state
        img_embeds = model.projector(img_feats)
        prefix_embeds = model.llama.get_input_embeddings()(prefix_ids.input_ids)
        asst_suffix=model.llama.get_input_embeddings()(suffix_ids.input_ids)
        
        # [Prompt Text] + [Visual Tokens]
        inputs_embeds = torch.cat([prefix_embeds, img_embeds,asst_suffix], dim=1)
        
        output_ids = model.llama.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    if os.path.exists(external_image_path):
        print(f"Image loaded: {external_image_path}")
        while True:
            query = input("\nAsk about the image (or 'exit'): ")
            if query.lower() == 'exit': break
            
            answer = chat_with_image(external_image_path, query)
            print(f"\nAI: {answer}")
    else:
        print("Image path not found. Please check external_image_path.")