import os
import shutil
import random

# Paths
base_dir = '../chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Ensure val subdirs exist
for class_name in ['NORMAL', 'PNEUMONIA']:
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Get image file names
    class_train_dir = os.path.join(train_dir, class_name)
    images = os.listdir(class_train_dir)

    # Shuffle and take 20%
    random.shuffle(images)
    val_count = int(0.2 * len(images))

    val_images = images[:val_count]

    for img in val_images:
        src_path = os.path.join(class_train_dir, img)
        dst_path = os.path.join(val_dir, class_name, img)
        shutil.move(src_path, dst_path)

    print(f"Moved {val_count} images from {class_name} to validation set.")
