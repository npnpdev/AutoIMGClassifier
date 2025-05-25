import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Constants
RAW_DIR = Path('data/raw')
PROC_DIR = Path('data/processed')
SPLITS_DIR = Path('data/splits')
IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}
RANDOM_SEED = 42


def prepare_directories():
    """
    Create processed and splits directories, clearing existing content.
    """

    # Ensure base directories exist and are empty
    for base in (PROC_DIR, SPLITS_DIR):
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)

    # Create subfolders for splits and classes
    for split in SPLIT_RATIOS:
        for cls in get_classes():
            (SPLITS_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def get_classes():
    """
    Return list of class folder names under RAW_DIR.
    """
    return [p.name for p in RAW_DIR.iterdir() if p.is_dir()]


def process_and_split():
    """
    Resize, normalize, and split images into train/val/test.
    """
    random.seed(RANDOM_SEED)

    for cls in get_classes():
        raw_class_dir = RAW_DIR / cls
        images = list(raw_class_dir.glob('*.*'))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS['train'])
        n_val = int(n_total * SPLIT_RATIOS['val'])
        # remaining goes to test
        n_test = n_total - n_train - n_val

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, img_list in splits.items():
            for img_path in img_list:
                save_image(img_path, cls, split)


def save_image(src_path: Path, cls: str, split: str):
    """
    Load image, resize, normalize pixel values, and save to processed and split directories.
    """
    try:
        img = Image.open(src_path).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Normalize by scaling pixel values to [0, 1]
        # Here we save normalized images by re-scaling back to [0,255] for JPEG/PNG storage,
        # actual normalization will occur at training time if needed.
        img_array = (np.array(img) / 255.0 * 255).astype('uint8')
        img_norm = Image.fromarray(img_array)

        # Save to processed folder
        proc_cls_dir = PROC_DIR / cls
        proc_cls_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_cls_dir / src_path.name
        img_norm.save(proc_path)

        # Copy to split folder
        dest_path = SPLITS_DIR / split / cls / src_path.name
        shutil.copy(proc_path, dest_path)

    except Exception as e:
        print(f"Error processing {src_path}: {e}")


if __name__ == '__main__':
    prepare_directories()
    process_and_split()
    print("Data preparation complete.")
