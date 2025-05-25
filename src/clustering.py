import os
import shutil
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

# Agglomerative Clustering on features
def run_clustering(features, n_clusters, linkage, affinity):
    # Create and fit the agglomerative clustering model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=affinity,
        linkage=linkage
    )
    return model.fit_predict(features)

# Save cluster labels to .npz file
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
    features_dir    = Path(cfg['FEATURES_DIR'])
    clusters_dir    = Path(cfg['CLUSTERS_DIR'])

    # Clear previous clusters if they exist
    if clusters_dir.exists():
        shutil.rmtree(clusters_dir)

    # Variables for file paths
    train_feats     = features_dir / cfg.get('TRAIN_FEATURES_FILE', 'train_features.npz')
    train_out       = clusters_dir / cfg.get('TRAIN_CLUSTERS_FILE', 'train_clusters.npz')
    test_feats      = features_dir / cfg.get('TEST_FEATURES_FILE', 'test_features.npz')
    test_out        = clusters_dir / cfg.get('TEST_CLUSTERS_FILE', 'test_clusters.npz')

    # Clustering parameters
    n_clusters = cfg.get('N_CLUSTERS', 3)
    linkage    = cfg.get('CLUSTER_LINKAGE', 'ward')
    affinity   = cfg.get('CLUSTER_AFFINITY', 'euclidean')

    # --- Train set clustering ---

    # Load train features and true labels
    train_features, train_labels = load_features(train_feats)
    
    # Run clustering on train set
    train_clusters = run_clustering(train_features, n_clusters, linkage, affinity)
    
    # Save train cluster labels alongside true labels
    save_cluster_labels(train_out, train_clusters, train_labels)
    print(f"Saved train cluster labels to {train_out}")

    # --- Test set clustering ---

    # Load test features (no true labels needed for visualization)
    test_features, _ = load_features(test_feats)
    
    # Run clustering on test set
    test_clusters = run_clustering(test_features, n_clusters, linkage, affinity)
    
    # Save test cluster labels
    save_cluster_labels(test_out, test_clusters)
    print(f"Saved test cluster labels to {test_out}")

if __name__ == '__main__':
    main()