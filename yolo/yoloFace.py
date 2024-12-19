import cv2
import numpy as np
from ultralytics import YOLO


COLOR = (0, 255, 0)

class YOLOFace:
    def __init__(self):
        self.net = YOLO("models/best.pt")
    def get_patches(self, img):
        patches = []
        for (x1, y1), (x2, y2) in self.forward(img):
            if (x1 == x2 or y1 == y2):
                continue
            patches.append(img[y1:y2, x1:x2])
        return patches
    def forward(self, img):
        boxes = self.net.predict(img, verbose=False)[0].boxes.xyxy.cpu().detach().numpy()
        for x1, y1, x2, y2 in boxes:
            yield (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item()))
    __call__ = get_patches
    
def display(yolo_model:YOLO):
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if not ret:
            print("unable to record")
            continue
        for (x1, y1), (x2, y2) in yolo_model.forward(img):
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
            cv2.putText(img, "face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
        cv2.imshow("Camera Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Show the image
    cam.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    yolo_model = YOLOFace()
    display(yolo_model=yolo_model)