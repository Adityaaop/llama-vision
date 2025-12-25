# import os
import torch
# from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor,AutoModel
# from hf_model import LlamaVisionModel

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
# def chat_with_vision(img_path, query):
#     raw_image = Image.open(img_path).convert("RGB")
#     pixel_values = image_processor(raw_image, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

#     prompt = query#f"{query}\nAssistant: <image>"
#     inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

#     with torch.no_grad():
#         img_feats = model.vision_encoder(pixel_values).last_hidden_state
#         img_embeds = model.projector(img_feats)
#         text_embeds = model.llama.get_input_embeddings()(inputs.input_ids)

#         inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)

#         output_ids = model.llama.generate(
#             inputs_embeds=inputs_embeds,
#             max_new_tokens=128,
#             do_sample=True,
#             temperature=0.8,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return response.strip()