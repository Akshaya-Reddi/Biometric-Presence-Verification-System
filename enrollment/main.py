from fastapi import FastAPI, UploadFile, File
import os
import uuid
import shutil
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet

app = FastAPI()
detector = MTCNN()
embedder = FaceNet()

TEMP_EMBEDDINGS = []

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_frames(video_path: str, video_id: str):
    cap = cv2.VideoCapture(video_path)

    frames_dir = os.path.join("uploads", "frames", video_id)
    os.makedirs(frames_dir, exist_ok=True)

    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Save every 5th frame (avoid too many images)
        if frame_count % 5 == 0:
            frame_path = os.path.join(
                frames_dir, f"frame_{frame_count}.jpg"
            )
            cv2.imwrite(frame_path, frame)

    cap.release()

def is_blurry(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def filter_good_frames(video_id: str):
    raw_dir = os.path.join("uploads", "frames", video_id)
    good_dir = os.path.join(raw_dir, "good")
    os.makedirs(good_dir, exist_ok=True)

    for filename in os.listdir(raw_dir):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(raw_dir, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[SKIP] Cannot read {filename}")
            continue

        if is_blurry(image):
            print(f"[REJECT BLUR] {filename}")
            continue

        h, w, _ = image.shape
        if h < 120 or w < 120:
            print(f"[REJECT SMALL IMAGE] {filename}")
            continue

        # Resize to stable size for MTCNN
        image_resized = cv2.resize(image, (320, 320))

        try:
            results = detector.detect_faces(image_resized)
        except Exception as e:
            print(f"[MTCNN ERROR] {filename} → {e}")
            continue

        face = results[0]
        if face['confidence'] < 0.85:
            print(f"[REJECT LOW CONF {face['confidence']:.2f}] {filename}")
            continue

        x, y, w, h = face['box']
        # Clamp bounding box to image boundaries
        img_h, img_w, _ = image.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Reject invalid crops
        if w <= 0 or h <= 0:
            print(f"[REJECT INVALID BOX] {filename}")
            continue

        if w < 60 or h < 60:
            print(f"[REJECT TOO SMALL FOR EMBEDDING] {filename}")
            continue

        save_path = os.path.join(good_dir, filename)
        cv2.imwrite(save_path, image)

        # Crop face
        x, y, w, h = face['box']
        face_img = image[y:y+h, x:x+w]

        # Resize for FaceNet
        face_img = cv2.resize(face_img, (160, 160))

        #Convert to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        #Generate embedding
        embedding = embedder.embeddings([face_img])[0]

        #Ensure correct shape
        if embedding.shape != (128,):
            print(f"[REJECT EMBEDDING SHAPE] {filename}")
            continue

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Store in memory (vector DB compatible format)
        TEMP_EMBEDDINGS.append({
            "vector": embedding,
            "metadata": {
                "video_id": video_id,
                "frame": filename,
                "confidence": face['confidence']
            }
        })

        print(f"[ACCEPTED + EMBEDDING STORED] {filename}")

@app.post("/enroll/upload-video")
async def upload_video(video: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    extract_frames(video_path, file_id)
    filter_good_frames(file_id)

    return {
        "status": "success",
        "message": "Frames extracted and filtered",
        "video_id": file_id
    }
