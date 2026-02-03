import os
import cv2
import numpy as np
from keras_facenet import FaceNet

# Initialize FaceNet once
embedder = FaceNet()

def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)

    try:
        embedding = embedder.embeddings([face_img])[0]
    except Exception as e:
        print(f"[EMBEDDING ERROR] {filename} → {str(e)}")
        continue

    return embedding[0]

def process_good_frames(video_id: str):
    good_dir = os.path.join(
        "uploads", "frames", video_id, "good"
    )

    embeddings = []

    for file in os.listdir(good_dir):
        if not file.endswith(".jpg"):
            continue

        img_path = os.path.join(good_dir, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        embedding = get_embedding(image)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")

    return embeddings


if __name__ == "__main__":
    VIDEO_ID = "1f8b97f0-446c-4206-8718-68466dce9ce9"
    process_good_frames(VIDEO_ID)
