import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
from model import LlamaVisionModel, MultimodalDataset

# Config
llama_name = "meta-llama/Llama-3.2-1B"
clip_name = "openai/clip-vit-large-patch14"
device = "cuda"

# Init
tokenizer = AutoTokenizer.from_pretrained(llama_name)
tokenizer.pad_token = tokenizer.eos_token
image_processor = CLIPImageProcessor.from_pretrained(clip_name)

model = LlamaVisionModel(llama_name, clip_name).to(device)
dataset = MultimodalDataset("./data", tokenizer, image_processor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.projector.parameters(), lr=2e-4)

print("Starting training...")
model.train()
for epoch in range(5):
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        
        pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        loss = model(pixel_values, input_ids, attention_mask)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")

torch.save(model.projector.state_dict(), "llama_vision_projector.pt")
print("Projector saved!")