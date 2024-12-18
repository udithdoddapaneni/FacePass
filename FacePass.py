from yolo.yolo import YOLO, display
from vgg.vgg_face import MODEL_FACE
from typing import Callable
import cv2

class VideoStream:
    def __init__(self):
        self.yolo_model = YOLO()
    def stream(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, img = cam.read()
            if not ret:
                print("unable to record")
                continue
            yield self.yolo_model(img)
            cv2.imshow("Camera Feed", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
    __call__ = stream

class FeatureExtractorStream:
    def __init__(self):
        self.model = MODEL_FACE
    def stream(self, video):
        for image in video:
            for patch in image:
                embeddings = self.model(patch)
                print(embeddings)
                yield embeddings
    __call__ = stream
class pipeline:
    def __init__(self):
        self.VideoStream = VideoStream()
        self.FeatureExtractorStream = FeatureExtractorStream()
    def forward(self):
        video = self.VideoStream()
        embeddings = self.FeatureExtractorStream(video)
        for embedding in embeddings:
            print(embedding)
    __call__ = forward

if __name__ == "__main__":
    pipe = pipeline()
    pipe()