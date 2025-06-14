import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # --------------------------------------------------
    # Settings and CWD
    # --------------------------------------------------
    # force CWD to the script's directory
    os.chdir(Path(__file__).resolve().parent)

    cfg_path = Path("config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    FEATURES_DIR = Path(cfg["FEATURES_DIR"])
    CLUSTERS_DIR = Path(cfg["CLUSTERS_DIR"])
    CLASS_DIR    = Path(cfg["CLASSIFICATION_DIR"])
    RAW_TEST_DIR = Path(cfg["SPLITS_DIR"]) / "test"
    ALLOWED_EXT  = set(cfg["ALLOWED_EXT"])

    # Variables for file paths
    features_path = FEATURES_DIR / cfg["TEST_FEATURES_FILE"]
    clusters_path = CLUSTERS_DIR / cfg["TEST_CLUSTERS_FILE"]
    preds_path    = CLASS_DIR  / cfg["TEST_PREDICTIONS_FILE"]

    # Destination directory for visualizations
    viz_dir = CLUSTERS_DIR / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Loading data
    # --------------------------------------------------
    data_feat   = np.load(features_path)
    y_true_text = data_feat["labels"]

    # Load image paths
    img_paths = data_feat["paths"]

    y_pred   = np.load(preds_path)["preds"]
    clusters = np.load(clusters_path)["clusters"]

    # Mapping class names to indices
    classes      = sorted(set(y_true_text))
    label_to_idx = {c:i for i,c in enumerate(classes)}
    y_true       = np.array([label_to_idx[t] for t in y_true_text])

    # --------------------------------------------------
    # Visualization of results
    # --------------------------------------------------
    print("\n=== Classification Report (Test Set) ===\n")
    print(classification_report(
        y_true, y_pred, target_names=classes,
        zero_division=0
    ))

    # --------------------------------------------------
    # Confusion Matrix (saved to file)
    # --------------------------------------------------
    # Create confusion matrix - how many true labels vs predicted
    cm = confusion_matrix(y_true, y_pred)

    # A figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')

    # Figure settings
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    cm_file = viz_dir / "confusion_matrix.png"
    plt.savefig(cm_file, dpi=200)
    print(f"Saved confusion matrix to {cm_file}")
    plt.close(fig)

    # --------------------------------------------------
    # Sample images with clusters and predictions
    # --------------------------------------------------
    # Choosing a random sample of images, max 9
    num_samples = min(9, len(img_paths))
    inds = np.random.choice(len(img_paths), num_samples, replace=False)

    # A grid of sample images
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Display images in the grid
    for ax, idx in zip(axes.flatten(), inds):
        img = Image.open(img_paths[idx]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Cluster {clusters[idx]} | Pred: {classes[y_pred[idx]]}",
            fontsize=10
        )

    plt.suptitle("Przykładowe obrazy testowe\nz klastrem i predykcją", y=0.99)
    plt.tight_layout()

    grid_file = viz_dir / "sample_grid.png"
    plt.savefig(grid_file, dpi=200)
    print(f"Saved sample grid to {grid_file}")
    plt.close(fig)

if __name__ == '__main__':
    main()