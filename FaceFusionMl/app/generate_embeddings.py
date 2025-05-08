import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

# Set paths
IMAGE_DIR = './images'  # Directory with student folders
EMBEDDING_OUTPUT = 'embeddings.json'  # Output embedding file
MODEL_PATH = 'facenet_model.pth'  # Path to save the model

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# To tensor
to_tensor = transforms.ToTensor()

# Where to store final results
embeddings = {}

# Loop through student folders
for student_id in tqdm(os.listdir(IMAGE_DIR), desc="Generating Embeddings"):
    student_folder = os.path.join(IMAGE_DIR, student_id)
    
    if not os.path.isdir(student_folder):
        continue  # Skip if not a folder

    print(f"\n👤 Processing student: {student_id}")
    
    student_embeddings = []

    for img_name in os.listdir(student_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(student_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(f"⚠️ Cannot open {img_name}")
            continue

        face = mtcnn(img)

        if face is None:
            print(f"❌ No face detected in {img_name}")
            continue

        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(face).squeeze().cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            student_embeddings.append(embedding)

        print(f"✅ Face detected and embedded from {img_name}")

    # Average the embeddings if any
    if student_embeddings:
        avg_embedding = np.mean(student_embeddings, axis=0)
        embeddings[student_id] = avg_embedding.tolist()
        print(f"🧠 Saved embedding for {student_id} from {len(student_embeddings)} images.")
    else:
        print(f"⚠️ No embeddings found for {student_id}")

# Save the JSON
with open(EMBEDDING_OUTPUT, 'w') as f:
    json.dump(embeddings, f, indent=2)

print(f"\n✅ Done! Saved {len(embeddings)} student embeddings to {EMBEDDING_OUTPUT}")

# Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")