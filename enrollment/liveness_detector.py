import cv2
import numpy as np
import sys
import os

# Add anti-spoof repo to path
ANTI_SPOOF_PATH = "Silent-Face-Anti-Spoofing"
sys.path.append(ANTI_SPOOF_PATH)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

model_dir = os.path.join(
    ANTI_SPOOF_PATH,
    "resources",
    "anti_spoof_models"
)

predictor = AntiSpoofPredict(0)
cropper = CropImage()

#Liveness Function
def check_liveness(frame):
    image = frame
    h, w, _ = image.shape

    prediction = np.zeros((1, 3))

    for model_name in os.listdir(model_dir):

        h_input, w_input, model_type, scale = parse_model_name(model_name)

        param = {
            "org_img": image,
            "bbox": [0, 0, w, h],
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }

        img = cropper.crop(**param)

        model_path = os.path.join(model_dir, model_name)

        prediction += predictor.predict(img, model_path)

    label = np.argmax(prediction)
    score = prediction[0][label] / 2

    # label 1 = real face
    is_live = label == 1

    return is_live, float(score)
