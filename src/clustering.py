import os
import numpy as np
import yaml
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering

# force CWD to the script's directory
os.chdir(Path(__file__).resolve().parent)

# Load configuration from config.yaml
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load features and labels from .npz file
def load_features(features_path):
    data = np.load(features_path)
    return data['features'], data['labels']

# Perform Agglomerative Clustering on features
def run_clustering(features, n_clusters, linkage, affinity):
    # Create and fit the agglomerative clustering model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=affinity,
        linkage=linkage
    )
    return model.fit_predict(features)

# Save cluster labels (and optional true labels) to .npz file
def save_cluster_labels(output_path, cluster_labels, true_labels=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if true_labels is not None:
        np.savez(output_path, clusters=cluster_labels, labels=true_labels)
    else:
        np.savez(output_path, clusters=cluster_labels)


def main():
    # Load config
    cfg = load_config()

    # Paths from config
    features_dir = Path(cfg['FEATURES_DIR'])
    clusters_dir = Path(cfg['CLUSTERS_DIR'])
    features_file = features_dir / cfg.get('TRAIN_FEATURES_FILE', 'train_features.npz')
    output_file = clusters_dir / cfg.get('TRAIN_CLUSTERS_FILE', 'train_clusters.npz')

    # Clustering parameters
    n_clusters = cfg.get('N_CLUSTERS', 3)
    linkage = cfg.get('CLUSTER_LINKAGE', 'ward')
    affinity = cfg.get('CLUSTER_AFFINITY', 'euclidean')

    # Load features
    features, labels = load_features(features_file)

    # Run clustering
    clusters = run_clustering(features, n_clusters, linkage, affinity)

    # Save results
    save_cluster_labels(output_file, clusters, labels)

    print(f"Saved cluster labels to {output_file}")

if __name__ == '__main__':
    main()