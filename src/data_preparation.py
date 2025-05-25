import shutil
import random
from pathlib import Path
from PIL import Image
import yaml
import os

# force CWD to the script's directory
os.chdir(Path(__file__).resolve().parent)

# Load configuration
with open("config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# Convert paths and settings
RAW_DIR = Path(cfg['RAW_DIR'])
SPLITS_DIR = Path(cfg['SPLITS_DIR'])
IMAGE_SIZE = tuple(cfg['IMAGE_SIZE'])
SPLIT_RATIOS = cfg['SPLIT_RATIOS']
RANDOM_SEED = cfg['RANDOM_SEED']
ALLOWED_EXT = set(cfg['ALLOWED_EXT'])

""" Create and clean directories for each dataset split. """
def prepare_directories():
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)
    for split in SPLIT_RATIOS:
        for cls in get_classes():
            (SPLITS_DIR / split / cls).mkdir(parents=True, exist_ok=True)

""" Return a list of class folder names from the raw data directory. """
def get_classes():
    return [p.name for p in RAW_DIR.iterdir() if p.is_dir()]

""" Resize images and split them into train/validation/test sets. """
def process_and_split():
    random.seed(RANDOM_SEED)

    for cls in get_classes():
        raw_class_dir = RAW_DIR / cls
        images = [p for p in raw_class_dir.iterdir()
                  if p.suffix.lower() in ALLOWED_EXT]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS['train'])
        n_val = int(n_total * SPLIT_RATIOS['val'])

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, img_list in splits.items():
            for img_path in img_list:
                save_image(img_path, cls, split)

""" Load an image, resize it, and save it to the appropriate split directory."""
def save_image(src_path: Path, cls: str, split: str):
    
    try:
        img = Image.open(src_path).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        dest_path = SPLITS_DIR / split / cls / src_path.name
        img.save(dest_path)
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")


if __name__ == '__main__':
    prepare_directories()
    process_and_split()
    print("Data preparation complete.")