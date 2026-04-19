import os
import cv2

input_root = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO"
output_root = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO_TILED"

tile_size = 640
overlap = 192
min_box_area = 8
min_box_visibility = 0.2

def make_dirs():
    os.makedirs(os.path.join(output_root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", "val"), exist_ok=True)

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(parts[0])
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        x1 = xc - bw / 2.0
        y1 = yc - bh / 2.0
        x2 = xc + bw / 2.0
        y2 = yc + bh / 2.0

        boxes.append([cls, x1, y1, x2, y2])

    return boxes

def clip_box_to_tile(box, tile_x1, tile_y1, tile_x2, tile_y2):
    cls, x1, y1, x2, y2 = box

    inter_x1 = max(x1, tile_x1)
    inter_y1 = max(y1, tile_y1)
    inter_x2 = min(x2, tile_x2)
    inter_y2 = min(y2, tile_y2)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1

    if inter_w <= 0 or inter_h <= 0:
        return None

    original_area = (x2 - x1) * (y2 - y1)
    clipped_area = inter_w * inter_h

    if original_area <= 0:
        return None

    visibility = clipped_area / original_area

    if clipped_area < min_box_area:
        return None

    if visibility < min_box_visibility:
        return None

    new_x1 = inter_x1 - tile_x1
    new_y1 = inter_y1 - tile_y1
    new_x2 = inter_x2 - tile_x1
    new_y2 = inter_y2 - tile_y1

    return [cls, new_x1, new_y1, new_x2, new_y2]

def write_yolo_labels(output_label_path, boxes, tile_w, tile_h):
    lines = []

    for box in boxes:
        cls, x1, y1, x2, y2 = box

        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0

        xc /= tile_w
        yc /= tile_h
        bw /= tile_w
        bh /= tile_h

        if bw <= 0 or bh <= 0:
            continue

        if xc < 0 or xc > 1 or yc < 0 or yc > 1:
            continue

        if bw > 1 or bh > 1:
            continue

        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(output_label_path, "w") as f:
        f.write("\n".join(lines))

def tile_split(split):
    image_dir = os.path.join(input_root, "images", split)
    label_dir = os.path.join(input_root, "labels", split)

    out_image_dir = os.path.join(output_root, "images", split)
    out_label_dir = os.path.join(output_root, "labels", split)

    step = tile_size - overlap

    image_files = os.listdir(image_dir)

    for image_name in image_files:
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        boxes = read_yolo_labels(label_path, img_w, img_h)

        tile_id = 0

        y = 0
        while y < img_h:
            x = 0
            while x < img_w:
                tile_x1 = x
                tile_y1 = y
                tile_x2 = min(x + tile_size, img_w)
                tile_y2 = min(y + tile_size, img_h)

                tile = image[tile_y1:tile_y2, tile_x1:tile_x2]

                tile_h_actual = tile_y2 - tile_y1
                tile_w_actual = tile_x2 - tile_x1

                new_boxes = []

                for box in boxes:
                    clipped = clip_box_to_tile(box, tile_x1, tile_y1, tile_x2, tile_y2)
                    if clipped is not None:
                        new_boxes.append(clipped)

                if len(new_boxes) > 0:
                    base_name = os.path.splitext(image_name)[0]
                    tile_image_name = f"{base_name}_tile_{tile_id}.jpg"
                    tile_label_name = f"{base_name}_tile_{tile_id}.txt"

                    out_image_path = os.path.join(out_image_dir, tile_image_name)
                    out_label_path = os.path.join(out_label_dir, tile_label_name)

                    cv2.imwrite(out_image_path, tile)
                    write_yolo_labels(out_label_path, new_boxes, tile_w_actual, tile_h_actual)

                tile_id += 1
                x += step

            y += step

def main():
    make_dirs()
    tile_split("train")
    tile_split("val")
    print("Tiling complete")

if __name__ == "__main__":
    main()