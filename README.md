# Bag of Visual Words (BoVW) Classifier

A Python implementation of the Bag of Visual Words algorithm for image classification. This project can be used to train BoVW models on image datasets using multiple feature extractors (SIFT, SURF, ORB, AKAZE, BRISK, KAZE).

## Features

- **Multiple Feature Extractors**: Support for SIFT, SURF, ORB, AKAZE, BRISK, and KAZE
- **Configurable Pipeline**: Fine-tune vocabulary size, descriptor sampling, image resizing, and classifier parameters
- **Spatial Pyramid Matching**: Multi-scale feature extraction for improved accuracy
- **Model Persistence**: Save and load trained models for reproducible inference
- **Comprehensive Metrics**: Accuracy, classification report, and confusion matrix

## Project Structure

```
├── bovw.py                         # Core BoVW classifier implementation
├── bovw_demo.ipynb                 # Basic training and evaluation demo
├── bovw_demo_large_dataset.ipynb   # Training and evaluation demo on larger dataset
├── bovw_save_load_demo.ipynb       # Model persistence demonstration
├── requirements.txt                # Python dependencies
├── caltech101_train_test_split.py  # Script for spliting dataset
├── artifacts/                      # Directory for saving trained models
├── images_all_reduced/             # Example dataset (from Caltech101)
└── images_all_balanced.zip         # Balanced Daltech101 dataset
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
model.fit(Path('images_all_reduced/train'))

# Evaluate on test set
metrics = model.evaluate(Path('images_all_reduced/test'))
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(metrics['classification_report'])

# Classify a single image
prediction = model.classify_image(Path('images_all_reduced/test/accordion/image.jpg'))
print(f"Prediction: {prediction}")

# Save the trained model
model.save('artifacts/my_model.pkl')

# Load the model later
loaded_model = BagOfVisualWordsClassifier.load('artifacts/my_model.pkl')
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

This project uses selected images form the Caltech101 dataset for training. To train the model on the whole datset download the dataset (https://data.caltech.edu/records/mzrjq-6wc02) and split it into train and test splits using the ```caltech101_train_test_split.py``` script:

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



## Advanced Usage

### Training with Different Feature Extractors

```python
from bovw import BagOfVisualWordsClassifier, BoVWConfig

for extractor in ['SIFT', 'ORB', 'AKAZE']:
    config = BoVWConfig(feature_extractor=extractor)
    model = BagOfVisualWordsClassifier(config)
    model.fit('images_all_reduced/train')
    metrics = model.evaluate('images_all_reduced/test')
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
    model.fit('images_all_reduced/train')
    metrics = model.evaluate('images_all_reduced/test')
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

