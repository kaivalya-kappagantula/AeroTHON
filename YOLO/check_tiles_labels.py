import os
import cv2
import random

image_dir = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO_TILED//images//train"
label_dir = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO_TILED//labels//train"

class_names = {
    0: "person",
    1: "automobile",
    2: "cycle"
}

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

random.shuffle(image_files)

for image_name in image_files[:20]:
    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")

    image = cv2.imread(image_path)
    if image is None:
        continue

    h, w = image.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc = float(parts[1]) * w
            yc = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)
 
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = class_names.get(cls, str(cls))
            cv2.putText(
                image,
                label_text,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow("Check Labels", image)
    key = cv2.waitKey(0)

    if key == 27:
        break

cv2.destroyAllWindows()