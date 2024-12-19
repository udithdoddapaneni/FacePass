import os
from database.utils import similarity
import cv2
from vgg.vgg_face import MODEL_FACE
from yolo.yoloFace import YOLO_FACE

class Retriever:
    """Base Retriever class"""
    def __init__(self, thres=0.8, folder_loc="database/images", *args, **kwargs):
        self.thres = thres
        self.folder_loc = folder_loc
    def unlock_lock(self, *args, **kwargs): ...
    def __call__(self, *args, **kwargs):
        return self.unlock_lock(*args, **kwargs)

class Naive(Retriever):
    def unlock_lock(self, emb):
        "Kind of Dynamic but very slow"
        for root, dirs, files in os.walk(self.folder_loc):
            for file in files:
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                for patch in YOLO_FACE(image):
                    embedding = MODEL_FACE(patch)
                    if similarity(emb, embedding) > self.thres:
                        return True
        return False
    
class BruteForceStore(Retriever):
    def __init__(self, *args, **kwargs):
        """
            Watch dog integration required later
            Only Use when the number of images are less
        """
        super().__init__(*args, **kwargs)
        self.embeddings = []
        for root, dirs, files in os.walk(self.folder_loc):
            for file in files:
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                for patch in YOLO_FACE(image):
                    embedding = MODEL_FACE(patch)
                    self.embeddings.append(embedding)

    def unlock_lock(self, emb):
        """Only Use when the number of images are less"""
        for embedding in self.embeddings:
            if similarity(emb, embedding) > self.thres:
                return True
        return False