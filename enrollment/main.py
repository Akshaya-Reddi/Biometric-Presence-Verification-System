from fastapi import FastAPI, UploadFile, File
import os
import uuid
import shutil
import cv2
from mtcnn import MTCNN
import numpy as np
from keras_facenet import FaceNet
import datetime
from vector_db import add_identity, search_identity
from liveness_detector import check_liveness

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

        if len(results) != 1:
            print(f"[REJECT FACE COUNT={len(results)}] {filename}")
            continue

        face = results[0]
        if face['confidence'] < 0.85:
            print(f"[REJECT LOW CONF {face['confidence']:.2f}] {filename}")
            continue

        x, y, w, h = face['box']
        # Clamp bounding box to image boundaries
        img_h, img_w, _ = image_resized.shape
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
        face_img = image_resized[y:y+h, x:x+w]

        # Resize for FaceNet
        face_img = cv2.resize(face_img, (160, 160))

        #Convert to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        #Generate embedding
        embedding = embedder.embeddings([face_img])

        #convert to flat vector safely
        embedding = np.array(embedding).squeeze()

        #Ensure correct shape & Validate FaceNet embedding
        if embedding.ndim != 1 or embedding.shape[0] != 512:
            print(f"[REJECT EMBEDDING SHAPE] {filename} → {embedding.shape}")
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

def save_identity(identity_data):
    db_path = "identity_store"
    os.makedirs(db_path, exist_ok=True)

    file_path = os.path.join(
        db_path,
        f"{identity_data['user_id']}.json"
    )

    with open(file_path, "w") as f:
        import json
        json.dump(identity_data, f, indent=4)

    print(f"[IDENTITY SAVED] {file_path}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def consolidate_embeddings():
    global TEMP_EMBEDDINGS

    if len(TEMP_EMBEDDINGS) < 10:
        print("[ENROLLMENT FAILED] Not enough raw embeddings")
        return None

    # ---- Step 4.1: Collect vectors ----
    vectors = [item["vector"] for item in TEMP_EMBEDDINGS]
    vectors = np.array(vectors)

    print(f"[QC] Raw embeddings: {len(vectors)}")

    # ---- Step 4.2: Remove near-duplicates ----
    unique_vectors = []

    for vec in vectors:
        if len(unique_vectors) == 0:
            unique_vectors.append(vec)
            continue

        similarities = [
            cosine_similarity(vec, uvec)
            for uvec in unique_vectors
        ]

        if max(similarities) < 0.95:
            unique_vectors.append(vec)
        else:
            print("[QC] Dropped near-duplicate")

    unique_vectors = np.array(unique_vectors)
    print(f"[QC] After deduplication: {len(unique_vectors)}")

    if len(unique_vectors) < 7:
        print("[ENROLLMENT FAILED] Too many duplicates")
        return None

    # ---- Step 4.3: Remove outliers ----
    mean_vector = np.mean(unique_vectors, axis=0)
    mean_vector = mean_vector / np.linalg.norm(mean_vector)

    final_vectors = []

    for vec in unique_vectors:
        sim = cosine_similarity(vec, mean_vector)
        if sim > 0.7:
            final_vectors.append(vec)
        else:
            print(f"[QC] Dropped outlier (sim={sim:.2f})")

    final_vectors = np.array(final_vectors)
    print(f"[QC] Final embeddings kept: {len(final_vectors)}")

    if len(final_vectors) < 5:
        print("[ENROLLMENT FAILED] Embeddings unstable")
        return None

    # ---- Step 4.4: Final identity vector ----
    identity_vector = np.mean(final_vectors, axis=0)
    identity_vector = identity_vector / np.linalg.norm(identity_vector)

    print("[IDENTITY CREATED] High-quality identity vector ready")

        # ---- Step 5: Stability Scoring ----
    stability_scores = [
        cosine_similarity(vec, identity_vector)
        for vec in final_vectors
    ]

    stability_score = float(np.mean(stability_scores))

    print(f"[STABILITY SCORE] {stability_score:.3f}")

    identity_package = {
        "user_id": str(uuid.uuid4()),   # temp user creation
        "identity_vector": identity_vector.tolist(),  # JSON safe
        "stability_score": stability_score,
        "embeddings_used": len(final_vectors),
        "enrollment_timestamp": datetime.datetime.utcnow().isoformat()
    }

    return identity_package

def select_best_match(matches, threshold=0.65):

    if not matches:
        return None

    # Remove FAISS invalid scores
    valid_matches = [
        m for m in matches if m["score"] > threshold
    ]

    if not valid_matches:
        return None

    # Sort by highest score
    best_match = sorted(
        valid_matches,
        key=lambda x: x["score"],
        reverse=True
    )[0]

    return best_match

def generate_embedding(frame):

    try:
        # Resize for stable detection
        frame_resized = cv2.resize(frame, (320, 320))

        results = detector.detect_faces(frame_resized)

        if len(results) != 1:
            return None

        face = results[0]

        if face["confidence"] < 0.85:
            return None

        x, y, w, h = face["box"]

        img_h, img_w, _ = frame_resized.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return None

        face_img = frame_resized[y:y+h, x:x+w]

        if face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (160, 160))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        embedding = embedder.embeddings([face_img])
        embedding = np.array(embedding).squeeze()

        if embedding.ndim != 1 or embedding.shape[0] != 512:
            return None

        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception:
        return None

MATCH_BUFFER_SIZE = 5  # number of frames to sample

def verify_multi_frame(frames):

    votes = {}
    confidence_map = {}

    for frame in frames:

        embedding = generate_embedding(frame)

        if embedding is None:
            continue

        matches = search_identity(embedding, top_k=1)
        best_match = select_best_match(matches)

        if best_match:

            uid = best_match["user_id"]
            score = best_match["score"]

            votes[uid] = votes.get(uid, 0) + 1

            if uid not in confidence_map:
                confidence_map[uid] = []

            confidence_map[uid].append(score)

    if not votes:
        return None, 0.0

    final_uid = max(votes, key=votes.get)

    stability = votes[final_uid] / len(frames)
    avg_confidence = float(np.mean(confidence_map[final_uid]))

    return final_uid, stability, avg_confidence

@app.post("/enroll/upload-video")
async def upload_video(video: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    extract_frames(video_path, file_id)
    filter_good_frames(file_id)

    identity_vector = consolidate_embeddings()

    if identity_vector is None:
        return {
            "status": "failed",
            "message": "Enrollment failed due to insufficient biometric data"
        }

    user_id = file_id
    add_identity(identity_vector["identity_vector"], user_id)


    return {
        "status": "success",
        "message": "Enrollment completed",
        "user_id": user_id,
        "stability_score": identity_vector["stability_score"]
    }

@app.post("/test/search")
async def test_search(video: UploadFile = File(...)):

    global TEMP_EMBEDDINGS
    TEMP_EMBEDDINGS = []

    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"test_{file_id}.mp4")

    # Save uploaded test video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Extract frames
    extract_frames(video_path, file_id)
    filter_good_frames(file_id)

    # Generate embeddings from TEMP storage
    if len(TEMP_EMBEDDINGS) == 0:
        return {"status": "failed", "message": "No embeddings generated"}

    # Average embeddings for test identity
    vectors = np.array([item["vector"] for item in TEMP_EMBEDDINGS])
    query_vector = np.mean(vectors, axis=0)
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search FAISS
    results = search_identity(query_vector)

    return {
        "status": "success",
        "matches": results
    }

@app.post("/attendance/verify")
async def verify_attendance(video: UploadFile = File(...)):

    #Save video
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"verify_{file_id}.mp4")

    # Save uploaded video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    #Read frames
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Sample every 5th frame
        if frame_count % 5 == 0:
            frames.append(frame)

        if len(frames) >= 5:
            break

    cap.release()

    if len(frames) == 0:
        return {"status": "failed", "message": "No valid frames captured"}

    #Liveness detection (multiframe)
    live_scores = []
    for f in frames:
        is_live, score = check_liveness(f)
        if is_live:
            live_scores.append(score)

    if not live_scores or np.mean(live_scores) < 0.5:
        return{
            "status": "spoof_detected",
            "liveness_score": float(np.mean(live_scores)) if live_scores else 0.0
        }
    
    #Identity verification
    # Multi-frame biometric verification
    user_id, stability, confidence = verify_multi_frame(frames)

    if user_id is None or stability < 0.6:
        return {"status": "no_match"}

    return {
        "status": "match",
        "user_id": user_id,
        "stability": stability,
        "confidence": confidence
    }
