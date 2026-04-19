import os
from PIL import Image
import shutil 

train_images_dir = r"C://Users//kaiva//Downloads//archive//VisDrone2019-DET-train//VisDrone2019-DET-train//images"
train_ann_dir = r"C://Users//kaiva//Downloads//archive//VisDrone2019-DET-train//VisDrone2019-DET-train//annotations"

val_images_dir = r"C://Users//kaiva//Downloads//archive//VisDrone2019-DET-val//VisDrone2019-DET-val//images"
val_ann_dir = r"C://Users//kaiva//Downloads//archive//VisDrone2019-DET-val//VisDrone2019-DET-val//annotations"

output_root = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO"

class_map = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:4,
    6:5,
    7:6,
    8:7,
    9:8,
    10:9
}

def convert_split(images_dir, ann_dir, split):
    image_files = os.listdir(images_dir)
    for img_name in image_files:
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(images_dir, img_name)
        ann_path = os.path.join(ann_dir, img_name.replace(".jpg",".txt"))
        img = Image.open(img_path)
        w, h = img.size
        yolo_lines = []
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split(",")
                x = float(parts[0])
                y = float(parts[1])
                bw = float(parts[2])
                bh = float(parts[3])
                score = int(parts[4])
                cls = int(parts[5])
                if score == 0:
                    continue
                if cls not in class_map:
                    continue
                x_center = (x + bw/2)/w
                y_center = (y + bh/2)/h
                bw = bw/w
                bh = bh/h
                yolo_cls = class_map[cls]
                yolo_lines.append(f"{yolo_cls} {x_center} {y_center} {bw} {bh}")
        out_img = os.path.join(output_root,"images",split,img_name)
        out_lbl = os.path.join(output_root,"labels",split,img_name.replace(".jpg",".txt"))
        shutil.copy(img_path,out_img)
        with open(out_lbl,"w") as f:
            f.write("\n".join(yolo_lines))

convert_split(train_images_dir,train_ann_dir,"train")
convert_split(val_images_dir,val_ann_dir,"val")

print("Conversion done")