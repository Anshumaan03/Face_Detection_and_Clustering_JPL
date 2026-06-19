import ssl
# Bypass local macOS SSL certificate restrictions if they persist
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

def verify_siglip2():
    print("🚀 Starting SigLIP 2 Verification...")
    
    # We use Google's base patch16 model configured for 256x256 resolution
    model_id = "google/siglip2-base-patch16-256"
    
    print(f"📥 Downloading/Loading weights for '{model_id}' from Hugging Face...")
    try:
        # Load the model and automatically place it on available hardware (CPU or Apple Silicon MPS)
        model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(model_id)
        print("✅ Model and Processor loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("\n⏳ Testing dummy image transformation and forward pass...")
    try:
        # Create a fake random RGB image (256x256) to test the pipeline without needing a real file yet
        random_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype('uint8'))
        
        # Preprocess the dummy image
        inputs = processor(images=[random_image], return_tensors="pt").to(model.device)
        
        # Generate the embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
            # Normalize the embedding vector (essential for Cosine Distance / Clustering later)
            normalized_embeddings = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
        print("✅ Embedding generated successfully!")
        print(f"📊 Vector Shape: {list(normalized_embeddings.shape)}") 
        print("💡 (Expected shape is [1, 768] — meaning 1 image yielded a 768-dimensional feature vector)\n")
        print("🎉 SigLIP 2 installation is perfectly working and ready!")
        
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")

if __name__ == "__main__":
    verify_siglip2()