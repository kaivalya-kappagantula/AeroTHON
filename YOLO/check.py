# import os

# label_dir = r"C://Users//kaiva//Projects//Aerothon//VisDrone_YOLO_TILED//labels//train"
# seen = set()

# for file_name in os.listdir(label_dir):
#     if file_name.endswith(".txt"):
#         with open(os.path.join(label_dir, file_name)) as f:
#             for line in f:
#                 seen.add(int(line.split()[0]))

# print(seen)


import os

label_dir = r"C:\Users\kaiva\Projects\Aerothon\VisDrone_YOLO_TILED\labels\train"
counts = {}

for file_name in os.listdir(label_dir):
    if not file_name.endswith(".txt"):
        continue

    with open(os.path.join(label_dir, file_name), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            counts[cls] = counts.get(cls, 0) + 1

print(counts)