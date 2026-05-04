# Bag of Visual Words (BoVW) Classifier

A Python implementation of the Bag of Visual Words algorithm for image classification and image retrieval. This project can be used to train BoVW models on image datasets using multiple feature extractors (SIFT, SURF, ORB, AKAZE, BRISK, KAZE).

## Features

- **Multiple Feature Extractors**: Support for SIFT, SURF, ORB, AKAZE, BRISK, and KAZE
- **Configurable Pipeline**: Fine-tune vocabulary size, descriptor sampling, image resizing, and classifier parameters
- **Spatial Pyramid Matching**: Multi-scale feature extraction for improved accuracy
- **Model Persistence**: Save and load trained models for reproducible inference
- **Image Search**: Build and persist a retrieval index, then query similar images with cosine similarity or Euclidean distance
- **Retrieval Metrics**: Precision@k, Recall@k, Average Precision (AP), and Mean Average Precision (mAP)
- **Classification Metrics**: Accuracy, classification report, and confusion matrix

## Project Structure

```
├── bovw.py                         # Core BoVW classifier implementation
├── bovw_demo.ipynb                 # Basic training and evaluation demo
├── bovw_save_load_demo.ipynb       # Model persistence demonstration
├── bovw_search_demo.ipynb          # Image search and retrieval metrics demo
├── requirements.txt                # Python dependencies
├── caltech101_train_test_split.py  # Script for splitting dataset
├── artifacts/                      # Directory for saving trained models
├── caltech101_reduced/             # Example dataset (from Caltech101)
└── caltech101_balanced.zip         # Balanced Caltech101 dataset
```

## Installation

1. **Clone the project directory**
   ```bash
   git clone https://github.com/KwiatkM/bag-of-visual-words-vs
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage in Python

```python
from pathlib import Path
from bovw import BagOfVisualWordsClassifier, BoVWConfig

# Create a config with desired parameters
config = BoVWConfig(
    feature_extractor='SIFT',      # Choose: SIFT, SURF, ORB, AKAZE, BRISK, KAZE
    vocabulary_size=128,            # Size of the visual vocabulary
    max_descriptors_per_image=250,  # Max descriptors to sample per image
    max_vocabulary_descriptors=20000,  # Max descriptors for vocabulary building
    random_state=42                 # For reproducibility
)

# Create and train the model
model = BagOfVisualWordsClassifier(config)
model.fit(Path('caltech101_reduced/train'))

# Evaluate on test set
metrics = model.evaluate(Path('caltech101_reduced/test'))
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(metrics['classification_report'])

# Classify a single image
prediction = model.classify_image(Path('caltech101_reduced/test/accordion/image.jpg'))
print(f"Prediction: {prediction}")

# Save the trained model
model.save('artifacts/my_model.pkl')

# Load the model later
loaded_model = BagOfVisualWordsClassifier.load('artifacts/my_model.pkl')

# Build an image search index and query similar images
test_images = sorted(
  p for p in Path('caltech101_reduced/test').rglob('*')
  if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
)
loaded_model.index_dataset(test_images, use_tfidf=True)
results = loaded_model.query(Path('caltech101_reduced/test/accordion/image_0010.jpg'), top_k=5)

# Evaluate retrieval quality with ranking metrics
metrics = loaded_model.evaluate_retrieval(
  [Path('caltech101_reduced/test/accordion/image_0010.jpg')],
  top_k=5,
)
print(metrics['mean_precision_at_k'], metrics['mean_recall_at_k'], metrics['mean_average_precision'])
```

## Configuration Parameters

The `BoVWConfig` dataclass controls all aspects of the pipeline:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_extractor` | str | `"SIFT"` | Feature detector to use. Options: `SIFT`, `SURF`, `ORB`, `AKAZE`, `BRISK`, `KAZE` |
| `vocabulary_size` | int | `128` | Number of visual words (clusters) in the vocabulary. Higher = more expressive but slower |
| `max_descriptors_per_image` | int | `250` | Maximum descriptors sampled per image during feature extraction |
| `max_vocabulary_descriptors` | int | `20000` | Maximum descriptors collected for vocabulary building across training set |
| `random_state` | int | `42` | Random seed for reproducibility |
| `classifier_c` | float | `1.0` | Regularization parameter for LinearSVC classifier |
| `resize_to` | int or None | `256` | Resize images so largest side = this value. `None` to skip resizing |
| `pyramid_levels` | tuple | `(1.0, 0.75, 0.5)` | Spatial pyramid scales for multi-scale feature extraction |

### Tuning Tips

- **Vocabulary Size**: Start with 128, increase to 256+ for more complex datasets
- **Feature Extractor**: 
  - `SIFT`: Most robust, slowest, requires OpenCV contrib
  - `SURF`: Fast and robust (if available)
  - `ORB`: Very fast, good for real-time applications
  - `AKAZE`: Free alternative to SIFT, good balance
- **Descriptors**: Increase `max_descriptors_per_image` for detailed images
- **Pyramid Levels**: More levels = better spatial info but slower computation

## Notebooks

### 1. `bovw_demo.ipynb`

Demonstrates the basic training and evaluation workflow:
- Import and configure the classifier
- Train on the training set
- Evaluate on the test set
- Display confusion matrix
- Show sample predictions with true/predicted labels



### 2. `bovw_save_load_demo.ipynb`

Demonstrates model persistence:
- Train a model
- Save it to disk
- Load it back
- Verify the loaded model predictions

### 3. `bovw_search_demo.ipynb`

Demonstrates image retrieval:
- Train the BoVW model
- Build and persist a retrieval index
- Query similar images with cosine or Euclidean distance
- Evaluate Precision@k, Recall@k, AP, and mAP across multiple queries

## Dataset Format

Images must be organized in the following structure:

```
dataset_root/
├── train/
│   ├── class_name_1/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ...
│   ├── class_name_2/
│   │   ├── image_1.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_name_1/
    │   ├── image_1.jpg
    │   └── ...
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`

This project uses selected images from the Caltech101 dataset for training. To train the model on the whole dataset, download the dataset (https://data.caltech.edu/records/mzrjq-6wc02) and split it into train and test splits using the ```caltech101_train_test_split.py``` script:

```bash
python caltech101_train_test_split.py [-h] [--test_ratio TEST_RATIO] [--seed SEED] [--balanced] dataset_dir output_dir
```

## API Reference

### `BagOfVisualWordsClassifier`

#### Methods

- **`__init__(config: BoVWConfig | None = None)`**
  - Initialize the classifier with optional configuration
  - Default config uses SIFT with 128 vocabulary size

- **`fit(train_dir: str | Path) -> BagOfVisualWordsClassifier`**
  - Train the model on a directory of training images
  - Images must be organized in subdirectories by class
  - Returns self for method chaining

- **`evaluate(test_dir: str | Path) -> dict`**
  - Evaluate on a test set directory
  - Returns dict with keys: `accuracy`, `classification_report`, `confusion_matrix`, `y_true`, `y_pred`, `image_paths`

- **`index_dataset(image_paths: Sequence[str | Path], use_tfidf: bool = False) -> None`**
  - Build an in-memory retrieval index from image paths
  - Optionally enable TF-IDF weighting for the indexed histograms

- **`query(query_image: str | Path, top_k: int = 5, metric: str = "cosine", use_tfidf: bool | None = None)`**
  - Retrieve the most similar images for a query image
  - Supports cosine similarity, Euclidean distance, and other pairwise metrics

- **`save_index(index_path: str | Path) -> Path`** / **`load_index(index_path: str | Path) -> BagOfVisualWordsClassifier`**
  - Save and restore the retrieval index from disk

- **`evaluate_retrieval(query_paths: Sequence[str | Path], top_k: int = 10, metric: str = "cosine", use_tfidf: bool | None = None, exclude_query: bool = True) -> dict`**
  - Evaluate retrieval quality for a list of queries
  - Returns per-query AP values plus mean Precision@k, mean Recall@k, and mAP


- **`predict(image_paths: Sequence[str | Path]) -> np.ndarray`**
  - Predict labels for multiple images
  - Returns array of predicted class names

- **`classify_image(image_path: str | Path) -> str`**
  - Predict label for a single image
  - Returns predicted class name as string

- **`extract_descriptors(image_path: str | Path) -> np.ndarray | None`**
  - Extract raw descriptors from an image
  - Returns array of descriptors or None if no features found

- **`save(model_path: str | Path) -> Path`**
  - Save trained model to disk
  - Automatically creates parent directories
  - Returns Path to saved model

- **`load(model_path: str | Path) -> BagOfVisualWordsClassifier`** (class method)
  - Load a previously saved model
  - Returns initialized classifier ready for inference

- **`supported_extractors() -> list[str]`** (static method)
  - Get list of available feature extractors for current OpenCV installation

- **`precision_at_k(...)`, `recall_at_k(...)`, `average_precision(...)`, `mean_average_precision(...)`**
  - Standalone helpers for ranking evaluation on binary relevance lists



## Advanced Usage

### Training with Different Feature Extractors

```python
from bovw import BagOfVisualWordsClassifier, BoVWConfig

for extractor in ['SIFT', 'ORB', 'AKAZE']:
    config = BoVWConfig(feature_extractor=extractor)
    model = BagOfVisualWordsClassifier(config)
    model.fit('caltech101_reduced/train')
    metrics = model.evaluate('caltech101_reduced/test')
    print(f"{extractor}: {metrics['accuracy']:.4f}")
```

### Hyperparameter Grid Search

```python
import itertools
from bovw import BagOfVisualWordsClassifier, BoVWConfig

vocab_sizes = [64, 128, 256]
extractors = ['SIFT', 'ORB', 'AKAZE']

results = []
for vocab_size, extractor in itertools.product(vocab_sizes, extractors):
    config = BoVWConfig(vocabulary_size=vocab_size, feature_extractor=extractor)
    model = BagOfVisualWordsClassifier(config)
    model.fit('caltech101_reduced/train')
    metrics = model.evaluate('caltech101_reduced/test')
    results.append({
        'extractor': extractor,
        'vocab_size': vocab_size,
        'accuracy': metrics['accuracy']
    })

# Find best configuration
best = max(results, key=lambda x: x['accuracy'])
print(f"Best: {best['extractor']} with vocab_size={best['vocab_size']}: {best['accuracy']:.4f}")
```

### Using Trained Models for Batch Inference

```python
from pathlib import Path
from bovw import BagOfVisualWordsClassifier

# Load pre-trained model
model = BagOfVisualWordsClassifier.load('artifacts/bovw_model.pkl')

# Predict on new images
test_images = list(Path('new_images').glob('*.jpg'))
predictions = model.predict(test_images)

# Print results
for img_path, pred in zip(test_images, predictions):
    print(f"{img_path.name}: {pred}")
```

### Image Search and Retrieval Metrics

```python
from pathlib import Path
from bovw import BagOfVisualWordsClassifier

model = BagOfVisualWordsClassifier.load('artifacts/bovw_model.pkl')
test_images = sorted(Path('caltech101_reduced/test').rglob('*.jpg'))

# Build and save the search index
model.index_dataset(test_images, use_tfidf=True)
model.save_index('artifacts/bovw_search_index.pkl')

# Query the index
query_path = test_images[0]
results = model.query(query_path, top_k=10, metric='cosine')

# Evaluate retrieval on multiple queries
metrics = model.evaluate_retrieval(test_images[:25], top_k=10, metric='cosine')
print(metrics['mean_precision_at_k'])
print(metrics['mean_recall_at_k'])
print(metrics['mean_average_precision'])
```

## Troubleshooting

### ImportError: No module named 'cv2'

Install OpenCV:
```bash
pip install opencv-contrib-python-headless
```

For full-featured OpenCV with GUI support:
```bash
pip install opencv-contrib-python
```

### SIFT/SURF not available

SIFT and SURF require the OpenCV contrib build. Install with:
```bash
pip install opencv-contrib-python
```

### AKAZE/ORB slower than expected

These feature extractors are inherently slower on large vocabularies. Try:
- Reducing `vocabulary_size`
- Reducing `max_descriptors_per_image`
- Using a GPU-accelerated OpenCV build

### Retrieval metrics look low

- Make sure the index was built with the same `use_tfidf` setting used at query time.
- Check that relevance is defined by class folder names, so queries and results must come from the same dataset split layout.
- Increase `top_k` if you want recall-oriented reporting over a larger slice of the ranking.

