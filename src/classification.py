import os
import shutil
import numpy as np
import yaml
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

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

# Encode textual labels to numeric indices
def encode_labels(labels):
    # Ensure labels are unique and sorted
    classes = sorted(set(labels))
    # Create a mapping from class names to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # Convert labels to indices
    idx_labels = np.array([class_to_idx[l] for l in labels])
    return idx_labels, class_to_idx

# Train k-NN classifier (with distance weighting)
def train_knn(train_X, train_y, n_neighbors, metric):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights='distance'    # closer neighbors have greater influence
    )
    # Fit the model to the training data
    model.fit(train_X, train_y)    
    return model

# Save model, scaler, and class mapping
def save_model_bundle(model, scaler, class_to_idx, output_path):
    # A bundle to save the model, scaler, and class mapping
    bundle = {
        'model': model,
        'scaler': scaler,
        'class_to_idx': class_to_idx
    }
    joblib.dump(bundle, output_path)

# Save predictions
def save_predictions(preds, true_labels, output_path):
    np.savez(output_path, preds=preds, true_labels=true_labels)

def main():
    # Load config
    cfg = load_config()

    # Paths and parameters from config
    features_dir = Path(cfg['FEATURES_DIR'])
    out_dir = Path(cfg.get('CLASSIFICATION_DIR', 'classification_results'))

    # Clean previous results
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Variables for file paths
    train_file = features_dir / cfg.get('TRAIN_FEATURES_FILE', 'train_features.npz')
    val_file   = features_dir / cfg.get('VAL_FEATURES_FILE', 'val_features.npz')
    test_file  = features_dir / cfg.get('TEST_FEATURES_FILE', 'test_features.npz')
    preds_file = cfg.get('TRAIN_PREDICTIONS_FILE', 'test_preds.npz')
    k          = cfg.get('K_NEIGHBORS', 5)
    metric     = cfg.get('KNN_METRIC', 'euclidean')

    # Load data
    X_train, y_train_text = load_features(train_file)
    X_val,   y_val_text   = load_features(val_file)
    X_test,  y_test_text  = load_features(test_file)

    # Encode labels
    y_train, class_to_idx = encode_labels(y_train_text)
    y_val   = np.array([class_to_idx[l] for l in y_val_text])
    y_test  = np.array([class_to_idx[l] for l in y_test_text])

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Train k-NN classifier
    model = train_knn(X_train, y_train, n_neighbors=k, metric=metric)

    # test
    test_preds = model.predict(X_test)

    # Save model bundle and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model_bundle(model, scaler, class_to_idx, out_dir / 'knn_bundle.joblib')
    save_predictions(test_preds, y_test_text, out_dir / preds_file)

    print(f"Model bundle and predictions saved to {out_dir}")

if __name__ == '__main__':
    main()