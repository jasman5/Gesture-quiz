import os
import random
import shutil


SOURCE_DIR = "data/data/gestures/alldata"
DEST_DIR = "data/data/gestures"

CLASSES = ["A", "B", "C", "D", "None"]
TRAIN_SPLIT = 0.8 

for split in ["train", "val"]:
    for c in CLASSES:
        os.makedirs(os.path.join(DEST_DIR, split, c), exist_ok=True)

for c in CLASSES:
    src_folder = os.path.join(SOURCE_DIR, c)
    imgs = os.listdir(src_folder)
    random.shuffle(imgs)

    train_count = int(len(imgs) * TRAIN_SPLIT)
    train_imgs = imgs[:train_count]
    val_imgs = imgs[train_count:]

    for img in train_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(DEST_DIR, "train", c, img))


    for img in val_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(DEST_DIR, "val", c, img))

    print(f"{c}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\n🎯 Dataset successfully split into train/val folders!")
