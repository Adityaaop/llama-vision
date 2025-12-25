import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import LlamaForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

# 1. SETUP MODELS
llama_name = "meta-llama/Llama-3.2-1B" # Replace with your choice
clip_name = "openai/clip-vit-large-patch14"

tokenizer = AutoTokenizer.from_pretrained(llama_name)
tokenizer.pad_token = tokenizer.eos_token
image_processor = CLIPImageProcessor.from_pretrained(clip_name)

def angular_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular distance between two vectors in radians.
    
    Args:
        u (torch.Tensor): First vector (1D or batched 2D tensor)
        v (torch.Tensor): Second vector (same shape as u)
    
    Returns:
        torch.Tensor: Angular distance in radians
    """
    # Ensure float type for precision
    u = u.float()
    v = v.float()
    u=u.squeeze(dim=1)
    v=v.squeeze(dim=1)
    # Normalize vectors to avoid division by zero
    u_norm = torch.norm(u, dim=-1, keepdim=True)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    
    if torch.any(u_norm == 0) or torch.any(v_norm == 0):
        raise ValueError("Zero-length vector detected; angular distance undefined.")
    
    # Cosine similarity
    cos_sim = torch.sum(u * v, dim=-1) / (u_norm.squeeze(-1) * v_norm.squeeze(-1))
    
    # Clamp to avoid numerical errors outside [-1, 1]
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # Angular distance in radians
    return torch.acos(cos_sim)

# 2. DATASET CLASS
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, image_processor, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        # Filter all unique filenames (removing extensions)
        self.filenames = list(set([f.split('.')[0] for f in os.listdir(data_dir)]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]
        
        # Load Image
        img_path = os.path.join(self.data_dir, f"{base_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Load Text Description
        json_path = os.path.join(self.data_dir, f"{base_name}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        description = data['prompt']

        # Tokenize Text
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
            "attention_mask": tokens.attention_mask.squeeze(0),
            "desc":description,
            "file":base_name
        }

# 3. THE ARCHITECTURE
class LlamaVisionTrainer(torch.nn.Module):
    def __init__(self, llama_name, clip_name,training=False):
        super().__init__()
        self.is_training=training
        self.llama = LlamaForCausalLM.from_pretrained(llama_name, dtype=torch.bfloat16)
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_name, dtype=torch.bfloat16)
        
        # Freeze both
        for p in self.llama.parameters(): p.requires_grad = False
        for p in self.vision_encoder.parameters(): p.requires_grad = False

        # Projector (The only trainable part)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.vision_encoder.config.hidden_size, self.llama.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(self.llama.config.hidden_size, self.llama.config.hidden_size)
        ).to(torch.bfloat16)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        img_feats = self.vision_encoder(pixel_values).last_hidden_state
        img_embeds = self.projector(img_feats)
        text_embeds = self.llama.get_input_embeddings()(input_ids)

        # Combine: [Image_Embeds] [Text_Embeds]
        inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
        
        # Create mask for combined sequence
        img_mask = torch.ones((img_embeds.shape[0], img_embeds.shape[1]), device=input_ids.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)

        # Labels: -100 for image tokens (so model doesn't try to predict them)
        # and input_ids for the text portion
        ignore_labels = torch.full((img_embeds.shape[0], img_embeds.shape[1]), -100, device=input_ids.device)
        full_labels = torch.cat([ignore_labels, input_ids], dim=1)

        # When training, this returns a loss automatically
        outputs = self.llama(
            inputs_embeds=inputs_embeds, 
            attention_mask=full_mask, 
            labels=full_labels
        )
        return outputs.loss
# 4. TRAINING LOOP
device = "cuda"
model = LlamaVisionTrainer(llama_name, clip_name,True).to(device)
dataset = MultimodalDataset("./data", tokenizer, image_processor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.AdamW(model.projector.parameters(), lr=2e-4) # Slightly higher LR for projector
def train():
    model.train()
    for epoch in range(5): # Increase epochs as needed
        for batch in loader:
            optimizer.zero_grad()
            
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # The model now returns the CrossEntropy loss between predicted and actual text
            loss = model(pixel_values, input_ids, attention_mask)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")




def infrence():
    model.eval()
    state = torch.load("llama_vision_projector.pt", map_location=device)
    model.projector.load_state_dict(state)

    def test_external_image(image_path,prefix_text = "What this image aboutS"):
        print(f"\nProcessing: {image_path}")
        
        # 1. Load and Preprocess the external image
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Use the same image_processor from your setup
        pixel_values = image_processor(raw_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)

        # 2. Setup the Prompt
        
        prefix_ids = tokenizer(prefix_text+"<image>", return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            # Get Image Embeddings through the Projector
            img_feats = model.vision_encoder(pixel_values).last_hidden_state
            img_embeds = model.projector(img_feats)
            
            # Get Tag Embeddings
            prefix_embeds = model.llama.get_input_embeddings()(prefix_ids.input_ids)
            
            # Combine [ <image> ] + [ Projected Image Features ]
            inputs_embeds = torch.cat([prefix_embeds, img_embeds], dim=1)
            
            # 3. Generate
            output_ids = model.llama.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Decode and Format
        description = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        clean_desc = description.replace("</image>", "").strip()
        
        print("-" * 30)
        print(f"<image>{clean_desc}</image>")
        print("-" * 30)

    # --- EXECUTION ---
    # Change this path to any image file on your PC
    external_image_path = r"C:\Users\pc\LLM\llama-vision\bahubali_poster.jpg" 

    if os.path.exists(external_image_path):
        while True:
            inp=input("Question: ")
            test_external_image(external_image_path,inp)
    else:
        print(f"File not found: {external_image_path}")
        print("Please update 'external_image_path' with a valid file path.")
    # --- REPLACEMENT END ---