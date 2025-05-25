[English](#english-version) | [Polska wersja](#polska-wersja)

---

## English Version

### Project Description

**AutoImageClassifier** is a Python-based pipeline for automatic image categorization into three classes—landscape, vehicle, and animal. It leverages transfer learning with MobileNetV2 for feature extraction, agglomerative clustering for unsupervised grouping, and k-Nearest Neighbors for supervised classification. The end-to-end workflow includes data preprocessing, feature extraction, clustering, classification, and result visualization.

### Key Features

* **Automatic Label Detection**: Automatically reads class labels from subfolders in `data/raw/`—no hardcoding needed (easily extendable beyond landscape, vehicle, and animal).

* **Configurable Formats**: Supported image extensions (`.jpg`, `.png`, etc.) are defined in `config.yaml` and fully customizable.

* **Adjustable Split Ratios**: Train/validation/test proportions can be tuned via the YAML configuration.

* **Data Preparation & Splitting**:

  * Automatic train/val/test split based on configurable ratios
  * Image resizing to 224×224 and pixel normalization

* **Feature Extraction**:

  * Pretrained MobileNetV2 (PyTorch) as a frozen feature extractor
  * Per-image transforms (resize, to-tensor, optional ImageNet normalization)

* **Unsupervised Clustering**:

  * Hierarchical agglomerative clustering (configurable linkage & affinity)
  * Outputs cluster assignments for train and test sets

* **Supervised Classification**:

  * k-NN with distance weighting
  * StandardScaler for feature normalization
  * Configurable `k` and distance metric

* **End-to-End Pipeline**:

  * Single-entry point `main.py` runs the full workflow (data prep → feature extraction → clustering → classification → visualization)
  * Individual scripts can also be run separately

* **Visualization & Reporting**:

  * Classification report (precision, recall, F1-score)
  * Confusion matrix saved as PNG
  * Sample image grid annotated with cluster ID and predicted label

### Technologies

* **Python 3.8+** – core language
* **PyTorch & torchvision** – feature extraction
* **scikit-learn** – clustering and classification
* **NumPy & PyYAML** – data handling and configuration
* **Pillow** – image I/O and resizing
* **Matplotlib** – plotting confusion matrix and sample grid

### Project Structure

```text
.
├── data/
│   ├── raw/                    # Raw images organized by category (e.g., animal, car, landscape)
│   ├── splits/                 # Generated train/val/test splits (organized by category)
│   └── features/              # Extracted features (.npz)
├── results/
│   ├── train_clusters.npz      # Clusters for training set
│   ├── test_clusters.npz       # Clusters for test set
│   ├── classification_results/
│   │   ├── knn_bundle.joblib   # k-NN model bundle
│   │   └── test_preds.npz      # Test set predictions
│   └── visualizations/
│       ├── confusion_matrix.png
│       └── sample_grid.png
├── src/
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── clustering.py
│   ├── classification.py
│   ├── visualization.py
│   ├── config.yaml
│   └── main.py
├── requirements.txt            # Dependency list
├── README.md                   # This documentation
└── LICENSE                     # License
```

> **Note:** The `splits/`, `features/`, and `results/` directories are automatically created when the pipeline is executed.

### Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/npnpdev/auto-image-classifier.git
   cd auto-image-classifier
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**:
   Place raw images in `data/raw/landscape`, `data/raw/vehicle`, `data/raw/animal`.

4. **Configure**:
   Edit `config.yaml` for paths, split ratios, supported formats, and model parameters.

5. **Run pipeline**:

   ```bash
   python src/main.py
   ```

---

## Polska wersja

### Opis projektu

**AutoImageClassifier** to pipeline w Pythonie do automatycznej kategoryzacji obrazów w trzech klasach: krajobraz, pojazd i zwierzę. Wykorzystuje transfer learning z MobileNetV2 do ekstrakcji cech, hierarchiczną klasteryzację aglomeracyjną do grupowania oraz k-Nearest Neighbors do klasyfikacji. Cały workflow obejmuje przygotowanie danych, ekstrakcję cech, klasteryzację, klasyfikację oraz wizualizację wyników.

### Kluczowe funkcje

* **Automatyczne wykrywanie klas**: Nazwy folderów w `data/raw/` są automatycznie traktowane jako etykiety (można dowolnie zmieniać klasy).

* **Konfigurowalne formaty**: Obsługiwane rozszerzenia plików graficznych (`.jpg`, `.png` itd.) można ustawić w `config.yaml`.

* **Dostosowywalny podział danych**: Proporcje zbiorów treningowego, walidacyjnego i testowego można ustawić w konfiguracji.

* **Przygotowanie i podział danych**:

  * Automatyczne dzielenie danych z `data/raw/` na `splits/` wg proporcji z konfiguracji
  * Zmiana rozmiaru do 224×224 i normalizacja

* **Ekstrakcja cech**:

  * MobileNetV2 jako zamrożony ekstraktor cech (PyTorch)
  * Transformacje obrazu (resize, tensor, opcjonalna normalizacja)

* **Nadzorowana klasyfikacja**:

  * Klasteryzacja hierarchiczna aglomeracyjna (linkage, affinity – konfiguracja)
  * Przypisania do klastrów zapisywane osobno dla train i test

* **Klasyfikacja k-NN**:

  * Skalowanie cech (StandardScaler)
  * Możliwość konfiguracji liczby sąsiadów i metryki

* **Pełna automatyzacja**:

  * Uruchomienie `main.py` automatycznie wykonuje wszystkie kroki pipeline'u
  * Można też uruchamiać pliki pojedynczo

* **Wizualizacja wyników**:

  * Raport klasyfikacji (precision, recall, F1)
  * Macierz pomyłek i siatka przykładowych obrazów zapisane w `results/visualizations/`

### Technologie

* **Python 3.8+**
* **PyTorch & torchvision**
* **scikit-learn**
* **NumPy & PyYAML**
* **Pillow**
* **Matplotlib**

### Struktura projektu

```text
.
├── data/
│   ├── raw/                    # Surowe dane (obrazy w podfolderach: animal, car, landscape)
│   ├── splits/                 # Zbiory train/val/test wygenerowane automatycznie
│   └── features/              # Wyekstrahowane cechy obrazów
├── results/
│   ├── train_clusters.npz      # Klastry dla treningu
│   ├── test_clusters.npz       # Klastry dla testu
│   ├── classification_results/
│   │   ├── knn_bundle.joblib   # Model k-NN
│   │   └── test_preds.npz      # Predykcje
│   └── visualizations/
│       ├── confusion_matrix.png
│       └── sample_grid.png
├── src/
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── clustering.py
│   ├── classification.py
│   ├── visualization.py
│   ├── config.yaml
│   └── main.py
├── requirements.txt            # Lista zależności
├── README.md                   # Dokumentacja
└── LICENSE                     # Licencja
```

> **Uwaga:** Foldery `splits/`, `features/` i `results/` generują się automatycznie po uruchomieniu pipeline’u.

### Sposób użycia

1. **Sklonuj repozytorium**:

   ```bash
   git clone https://github.com/npnpdev/auto-image-classifier.git
   cd auto-image-classifier
   ```

2. **Zainstaluj zależności**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Umieść dane**:
   Wgraj obrazy do folderów: `data/raw/landscape`, `data/raw/vehicle`, `data/raw/animal`

4. **Skonfiguruj projekt**:
   Dostosuj `config.yaml`: proporcje podziału, wspierane formaty, parametry modeli

5. **Uruchom pipeline**:

   ```bash
   python src/main.py
   ```

---

## Autor / Author

Igor Tomkowicz

📧 [npnpdev@gmail.com](mailto:npnpdev@gmail.com)

GitHub: [npnpdev](https://github.com/npnpdev)

LinkedIn: [https://www.linkedin.com/in/igor-tomkowicz-a5760b358/](https://www.linkedin.com/in/igor-tomkowicz-a5760b358/

---

## Licencja / License

MIT License. Szczegóły w pliku [LICENSE](LICENSE).
