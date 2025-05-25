import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Force CWD to script directory
os.chdir(Path(__file__).resolve().parent)

# Load config
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Load features and clustering/classification outputs
def load_npz(path):
    data = np.load(path)
    return dict(data)

# Load and preprocess image for display
def load_image(image_path):
    return Image.open(image_path)

# Reduce dimensions for visualization
def reduce_dimensions(features, method='pca', n_components=2):
    reducer = PCA(n_components=n_components) if method == 'pca' else TSNE(n_components=n_components)
    return reducer.fit_transform(features)

# Plot samples with cluster and predicted class labels
def plot_visualization(points, images, cluster_labels, pred_labels, output_dir):
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(points[:, 0], points[:, 1], c=pred_labels, cmap='tab10', alpha=0.6, edgecolor='k')

    for i, img_path in enumerate(images):
        img = load_image(img_path).resize((24, 24))
        ax.imshow(img, extent=(points[i, 0]-1, points[i, 0]+1, points[i, 1]-1, points[i, 1]+1), zorder=2)

    legend = ax.legend(*scatter.legend_elements(), title="Predicted Class")
    ax.add_artist(legend)
    ax.set_title("Feature Visualization with Cluster and Prediction Labels")

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "visualization.png", dpi=300)
    plt.close()

def main():
    cfg = load_config()

    features_dir = Path(cfg['FEATURES_DIR'])
    clusters_dir = Path(cfg['CLUSTERS_DIR'])
    raw_images_dir = Path(cfg['SPLITS_DIR']) / 'test'
    output_dir = clusters_dir / "visualizations"

    # File paths
    features_path = features_dir / 'test_features.npz'
    clusters_path = clusters_dir / 'train_clusters.npz'
    preds_path = clusters_dir / 'classification_results' / 'test_preds.npz'

    # Load data
    features_data = load_npz(features_path)
    clusters_data = load_npz(clusters_path)
    preds_data = load_npz(preds_path)

    features = features_data['features']
    labels = features_data['labels']
    clusters = clusters_data['clusters']
    preds = preds_data['preds']

    # Find corresponding images from raw directory
    all_images = sorted([
        p for p in raw_images_dir.glob('**/*')
        if p.suffix.lower() in cfg['ALLOWED_EXT']
    ])

    if len(all_images) != len(features):
        print("Warning: Number of images and features do not match. Using subset.")
        all_images = all_images[:len(features)]

    # Reduce features to 2D
    points = reduce_dimensions(features, method='pca')

    # Generate visualization
    plot_visualization(points, all_images, clusters, preds, output_dir)

    print(f"Saved visualization to {output_dir/'visualization.png'}")

if __name__ == '__main__':
    main()
