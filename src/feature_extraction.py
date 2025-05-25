import os
import yaml
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np

# force CWD to the script's directory
os.chdir(Path(__file__).resolve().parent)

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def prepare_model():
    # Load pretrained MobileNetV2 and remove classifier
    model = models.mobilenet_v2(pretrained=True)
    # Keep feature extractor: all layers except classifier
    feature_extractor = torch.nn.Sequential(
        model.features,
        torch.nn.AdaptiveAvgPool2d((1, 1))
    )
    feature_extractor.eval()
    return feature_extractor


def extract_features(image_path, transform, model, device):
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x)
    # feat shape: [1, 1280, 1, 1]
    feat = feat.view(feat.size(0), -1).cpu().numpy()
    return feat.squeeze()


def main():
    cfg = load_config()
    splits_dir = Path(cfg['SPLITS_DIR'])
    output_dir = Path(cfg.get('FEATURES_DIR', 'features'))
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model().to(device)

    # Transform: normalize to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for split in ['train', 'val', 'test']:
        features = []
        labels = []
        paths = []
        split_dir = splits_dir / split
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in cfg['ALLOWED_EXT']:
                    continue
                try:
                    feat = extract_features(img_path, transform, model, device)
                    features.append(feat)
                    labels.append(label)
                    paths.append(str(img_path))   # collecting paths for later use
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        features = np.stack(features)
        labels = np.array(labels)
        paths = np.array(paths)
        np.savez(output_dir / f"{split}_features.npz",
            features=features,
            labels=labels,
            paths=paths)     
        print(f"Saved features for {split}: {features.shape}")

        # Clear memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
