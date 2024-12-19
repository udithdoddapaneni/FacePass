import cv2
import numpy as np

# need to change these offsets later
X1_OFFSET, X2_OFFSET = 0,0
Y1_OFFSET, Y2_OFFSET = 0,0

class YOLO:
    def __init__(self):
        self.net = cv2.dnn.readNet("models/yolov3-tiny.weights", "configs/yolov3-tiny.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
    def get_patches(self, img):
        patches = []
        for (x1, y1), (x2, y2), color, confidence, label in self.forward(img):
            if (x1 == x2 or y1 == y2):
                continue
            print((x1, y1), (x2, y2))
            patches.append(img[y1:y2, x1:x2])
        return patches
    def forward(self, img):
        height, width, channels = img.shape

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        # Run the forward pass
        outs = self.net.forward(self.output_layers)

        # Processing the output
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:] # center x, center y, width,  height, object confidence score, class confidence scores...
                class_id = np.argmax(scores)
                class_confidence = scores[class_id]
                object_confidence = detection[4]
                if object_confidence > 0.5:
                    # Get the coordinates for the bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    if x < 0 or y < 0:
                        continue
                    boxes.append([x, y, w, h])
                    confidences.append(float(class_confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green box
                if label == "person":
                    yield (x + X1_OFFSET, y + Y1_OFFSET), (x + w + X2_OFFSET, y + h + Y2_OFFSET), color, confidence, label
    __call__ = get_patches
    
def display(yolo_model:YOLO):
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if not ret:
            print("unable to record")
            continue
        for (x1, y1), (x2, y2), color, confidence, label in yolo_model.forward(img):
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Camera Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Show the image
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    yolo_model = YOLO()
    display(yolo_model=yolo_model)