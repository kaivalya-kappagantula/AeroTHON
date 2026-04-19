import os

label_root = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO_TILED//labels"

remap = {
    0: 0,  # pedestrian -> person
    1: 0,  # people -> person
    2: 2,  # bicycle -> cycle
    3: 1,  # car -> automobile
    4: 1,  # van -> automobile
    5: 1,  # truck -> automobile
    6: 2,  # tricycle -> cycle
    7: 2,  # awning-tricycle -> cycle
    8: 1,  # bus -> automobile
    9: 2   # motor -> cycle
}

for split in ["train", "val"]:
    split_dir = os.path.join(label_root, split)

    for file_name in os.listdir(split_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(split_dir, file_name)
        new_lines = []

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            old_cls = int(parts[0])
            if old_cls not in remap:
                continue

            new_cls = remap[old_cls]
            new_lines.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("General class merging complete.")