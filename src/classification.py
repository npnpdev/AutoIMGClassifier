import os
import numpy as np
import yaml
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# force CWD to the script's directory
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass  # __file__ not defined (e.g. in notebooks)

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
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_labels = np.array([class_to_idx[l] for l in labels])
    return idx_labels, class_to_idx

# Train k-NN classifier (with distance weighting)
def train_knn(train_X, train_y, n_neighbors, metric):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights='distance'    # closer neighbors have greater influence
    )
    model.fit(train_X, train_y)
    return model

# Evaluate model and print report
def evaluate_model(model, X, y, split_name):
    preds = model.predict(X)
    print(f"--- {split_name} Classification Report ---")
    print(classification_report(y, preds))
    print(f"--- {split_name} Confusion Matrix ---")
    print(confusion_matrix(y, preds))
    return preds

# Save model, scaler, and class mapping
def save_model_bundle(model, scaler, class_to_idx, output_path):
    bundle = {
        'model': model,
        'scaler': scaler,
        'class_to_idx': class_to_idx
    }
    joblib.dump(bundle, output_path)

# Save predictions
def save_predictions(preds, output_path):
    np.savez(output_path, preds=preds)

def main():
    # Load config
    cfg = load_config()

    # Paths and parameters from config
    features_dir = Path(cfg['FEATURES_DIR'])
    out_dir = Path(cfg.get('CLASSIFICATION_DIR', 'classification_results'))
    train_file = features_dir / cfg.get('TRAIN_FEATURES_FILE', 'train_features.npz')
    val_file   = features_dir / cfg.get('VAL_FEATURES_FILE', 'val_features.npz')
    test_file  = features_dir / cfg.get('TEST_FEATURES_FILE', 'test_features.npz')
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

    # Evaluate
    val_preds = evaluate_model(model, X_val, y_val, 'Validation')
    test_preds = evaluate_model(model, X_test, y_test, 'Test')

    # Save model bundle and predictions
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model_bundle(model, scaler, class_to_idx, out_dir / 'knn_bundle.joblib')
    save_predictions(test_preds, out_dir / 'test_preds.npz')

    print(f"Model bundle and predictions saved to {out_dir}")

if __name__ == '__main__':
    main()