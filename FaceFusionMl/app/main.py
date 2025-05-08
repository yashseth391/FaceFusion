from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import json
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import shutil

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition using FaceNet and MTCNN",
    version="1.0.0"
)

# Constants
EMBEDDINGS_PATH = './app/embeddings.json'  # Path to the stored embeddings
SIMILARITY_THRESHOLD = 0.7  # Threshold for recognizing a person (Adjustable)
MODEL_PATH = "./app/facenet_model.pth"
TEMP_UPLOAD_DIR = "./app/temp_uploads"

# Ensure temp directory exists
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Model
print("\nLoading model...")
model = InceptionResnetV1(pretrained=None).to(device).eval()
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if "logits" not in k}
    model.load_state_dict(filtered_state_dict)
    print("Model loaded successfully.")
except (RuntimeError, FileNotFoundError) as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MTCNN for Face Detection
print("\nInitializing MTCNN...")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

# Load Stored Embeddings
try:
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")
    
    with open(EMBEDDINGS_PATH, 'r') as f:
        stored_embeddings = json.load(f)
    print(f"Loaded {len(stored_embeddings)} embeddings from {EMBEDDINGS_PATH}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Face Recognition Function
def recognize_faces(image_path):
    """Process an image and return recognized faces"""
    try:
        img = Image.open(image_path).convert('RGB')  # Open and convert to RGB
        faces = mtcnn(img)  # Detect and align faces
        
        if faces is None or len(faces) == 0:
            return {"error": "No faces detected in the image."}
        
        # Ensure faces is a list even when only one face is detected
        if faces.ndim == 3:
            faces = [faces]
            
        # Generate embeddings for all detected faces
        embeddings = []
        for face in faces:
            face = face.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                embedding = model(face).squeeze().cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        
        # Identify Faces
        identified_faces = []
        for embedding in embeddings:
            best_match = None
            best_score = -1  # Cosine similarity ranges from -1 to 1
            # Compare with stored embeddings
            for student_id, stored_embedding in stored_embeddings.items():
                stored_embedding = np.array(stored_embedding)
                score = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
                if score > best_score:
                    best_score = score
                    best_match = student_id
            
            if best_score < SIMILARITY_THRESHOLD:
                identified_faces.append({
                    "identity": "Unknown", 
                    "confidence": None,
                    "similarity_score": float(best_score) if best_score > -1 else None
                })
            else:
                identified_faces.append({
                    "identity": best_match, 
                    "confidence": float(best_score * 100),
                    "similarity_score": float(best_score)
                })
        
        return {"faces": identified_faces}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running", "status": "active"}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Upload an image and get recognized faces
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, unique_filename)
    
    try:
        # Save uploaded file temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image
        result = recognize_faces(temp_file_path)
        
        # Return the result
        if "error" in result:
            return JSONResponse(content={"error": result["error"]}, status_code=400)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/students")
async def get_students():
    """Return a list of all registered students"""
    try:
        return {"students": list(stored_embeddings.keys())}
    except Exception as e:
        return {"error": str(e)}