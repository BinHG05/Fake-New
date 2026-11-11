import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

def main():
    print("="*60)
    print("🔍 DEMO: TEXT-IMAGE MISMATCH DETECTION USING OPENAI CLIP")
    print("="*60)
    
    print("\n[1/3] Loading CLIP Model (openai/clip-vit-base-patch32) - May take a moment...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    
    file_path = "data/02_processed/Reddit/dataset_Reddit_reddit_run_013.jsonl"
    print(f"\n[2/3] Reading pipeline data from {file_path}...")
    
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3: # Display 3 records
                    break
                records.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Cannot find {file_path}. Please make sure you are in the project root.")
        return
            
    print("\n[3/3] Running Consistency Analysis...")
    for i, data in enumerate(records):
        print(f"\n" + "-"*50)
        print(f"📄 Sample {i+1} | ID: {data.get('id', 'N/A')}")
        
        text = data.get("raw_text", "")
        print(f"📝 Text: {text}")
        
        img_info = data.get("image_info", {})
        img_path = img_info.get("processed_path", "")
        img_path = img_path.replace("\\", "/") # Normalize path
        
        if not img_path or not os.path.exists(img_path):
            print(f"🚫 Info: Image file not found locally ({img_path}). Skipping...")
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Truncate text to 77 tokens as required by CLIP
            inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract embeddings
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize to get cosine similarity (0 to 1 scale typically, mathematically -1 to +1)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            similarity = (image_embeds @ text_embeds.T).item()
            
            # Output Results
            print(f"📊 CLIP Cosine Similarity: {similarity:.4f}")
            if similarity < 0.22:
                print("🚨 RESULT: LOW CONSISTENCY -> Mismatch Detected! High probability of Fake News (Unrelated Image).")
            else:
                print("✅ RESULT: HIGH CONSISTENCY -> Text and Image are semantically aligned.")
                
        except Exception as e:
            print(f"Error processing image: {e}")

    print("\n" + "="*60)
    print("✅ Analysis Complete.")
    print("="*60)

if __name__ == "__main__":
    main()
