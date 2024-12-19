import os
import numpy as np
import cv2
from vgg.vgg_face import MODEL_FACE

def similarity(v1:np.ndarray, v2:np.ndarray):
    num = (v1 @ v2.T).item()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num/denom

def unlock_lock(emb, thres=0.6, folder_loc="database/images"):
    for root, dirs, files in os.walk(folder_loc):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path)
            embedding = MODEL_FACE(image)
            if similarity(emb, embedding) > thres:
                return True
    return False