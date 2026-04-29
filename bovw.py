from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class BoVWConfig:
    feature_extractor: str = "SIFT"
    vocabulary_size: int = 128
    max_descriptors_per_image: int = 250
    max_vocabulary_descriptors: int = 20000
    random_state: int = 42
    classifier_c: float = 1.0
    resize_to: int | None = 256
    pyramid_levels: Sequence[float] = (1.0, 0.75, 0.5)


class BagOfVisualWordsClassifier:
    def __init__(self, config: BoVWConfig | None = None):
        self.config = config or BoVWConfig()
        self.extractor = self._create_extractor(self.config.feature_extractor)
        self.kmeans: MiniBatchKMeans | None = None
        self.classifier: LinearSVC | None = None
        self.class_names: list[str] = []
        self._rng = np.random.default_rng(self.config.random_state)

    @staticmethod
    def supported_extractors() -> list[str]:
        extractors = ["SIFT", "ORB", "AKAZE", "BRISK", "KAZE"]
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
            extractors.insert(1, "SURF")
        return extractors

    @staticmethod
    def _create_extractor(name: str):
        normalized = name.upper().strip()
        if normalized == "SIFT":
            if not hasattr(cv2, "SIFT_create"):
                raise RuntimeError("SIFT is not available in the installed OpenCV build.")
            return cv2.SIFT_create()
        if normalized == "SURF":
            if not (hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create")):
                raise RuntimeError("SURF is not available in the installed OpenCV build. Install an OpenCV contrib build with nonfree features enabled.")
            return cv2.xfeatures2d.SURF_create()
        if normalized == "ORB":
            return cv2.ORB_create(nfeatures=2000)
        if normalized == "AKAZE":
            return cv2.AKAZE_create()
        if normalized == "BRISK":
            return cv2.BRISK_create()
        if normalized == "KAZE":
            return cv2.KAZE_create()
        raise ValueError(f"Unsupported feature extractor: {name}")

    def _preprocess_image(self, image_path: str | Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        if self.config.resize_to is not None:
            height, width = image.shape[:2]
            largest_side = max(height, width)
            if largest_side > self.config.resize_to:
                scale = self.config.resize_to / float(largest_side)
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def _extract_descriptors_from_image(self, image: np.ndarray) -> np.ndarray | None:
        all_descriptors: list[np.ndarray] = []
        for scale in self.config.pyramid_levels:
            if scale == 1.0:
                scaled = image
            else:
                new_width = max(1, int(round(image.shape[1] * scale)))
                new_height = max(1, int(round(image.shape[0] * scale)))
                scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            keypoints, descriptors = self.extractor.detectAndCompute(scaled, None)
            if descriptors is not None and len(descriptors) > 0:
                descriptors = descriptors.astype(np.float32, copy=False)
                all_descriptors.append(descriptors)
        if not all_descriptors:
            return None
        if len(all_descriptors) == 1:
            descriptors = all_descriptors[0]
        else:
            descriptors = np.vstack(all_descriptors)
        if len(descriptors) > self.config.max_descriptors_per_image:
            indices = self._rng.choice(len(descriptors), size=self.config.max_descriptors_per_image, replace=False)
            descriptors = descriptors[indices]
        return descriptors.astype(np.float32, copy=False)

    def extract_descriptors(self, image_path: str | Path) -> np.ndarray | None:
        image = self._preprocess_image(image_path)
        return self._extract_descriptors_from_image(image)

    def _iter_image_files(self, split_dir: str | Path) -> tuple[list[Path], list[str]]:
        split_dir = Path(split_dir)
        image_paths: list[Path] = []
        labels: list[str] = []
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                    image_paths.append(image_path)
                    labels.append(class_dir.name)
        if not image_paths:
            raise ValueError(f"No image files found in {split_dir}")
        return image_paths, labels

    def _sample_vocabulary_descriptors(self, image_paths: Sequence[Path]) -> np.ndarray:
        collected: list[np.ndarray] = []
        total = 0
        for image_path in image_paths:
            descriptors = self.extract_descriptors(image_path)
            if descriptors is None or len(descriptors) == 0:
                continue
            if total + len(descriptors) > self.config.max_vocabulary_descriptors:
                remaining = self.config.max_vocabulary_descriptors - total
                if remaining <= 0:
                    break
                descriptors = descriptors[:remaining]
            collected.append(descriptors)
            total += len(descriptors)
            if total >= self.config.max_vocabulary_descriptors:
                break
        if not collected:
            raise ValueError("No descriptors could be extracted from the training set.")
        vocabulary = np.vstack(collected).astype(np.float32, copy=False)
        if len(vocabulary) < self.config.vocabulary_size:
            raise ValueError(
                f"Not enough descriptors ({len(vocabulary)}) to build a vocabulary of size {self.config.vocabulary_size}."
            )
        return vocabulary

    def fit(self, train_dir: str | Path):
        train_paths, train_labels = self._iter_image_files(train_dir)
        self.class_names = sorted(set(train_labels))

        vocabulary_descriptors = self._sample_vocabulary_descriptors(train_paths)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.config.vocabulary_size,
            random_state=self.config.random_state,
            batch_size=max(100, self.config.vocabulary_size * 3),
            n_init="auto",
        )
        self.kmeans.fit(vocabulary_descriptors)

        train_features = self.transform(train_paths)
        self.classifier = LinearSVC(C=self.config.classifier_c, random_state=self.config.random_state)
        self.classifier.fit(train_features, train_labels)
        return self

    def _image_histogram(self, descriptors: np.ndarray | None) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("The vocabulary has not been fitted yet.")
        histogram = np.zeros(self.config.vocabulary_size, dtype=np.float32)
        if descriptors is None or len(descriptors) == 0:
            return histogram
        words = self.kmeans.predict(descriptors.astype(np.float32, copy=False))
        for word in words:
            histogram[word] += 1.0
        histogram = normalize(histogram.reshape(1, -1), norm="l2")[0].astype(np.float32, copy=False)
        return histogram

    def transform(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        if self.kmeans is None:
            raise RuntimeError("Call fit() before transform().")
        histograms = []
        for image_path in image_paths:
            descriptors = self.extract_descriptors(image_path)
            histograms.append(self._image_histogram(descriptors))
        return np.vstack(histograms)

    def predict(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Call fit() before predict().")
        features = self.transform(image_paths)
        return self.classifier.predict(features)

    def evaluate(self, test_dir: str | Path) -> dict[str, object]:
        if self.classifier is None:
            raise RuntimeError("Call fit() before evaluate().")
        test_paths, test_labels = self._iter_image_files(test_dir)
        predictions = self.predict(test_paths)
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, zero_division=0)
        matrix = confusion_matrix(test_labels, predictions, labels=self.class_names)
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix,
            "y_true": np.array(test_labels),
            "y_pred": predictions,
            "image_paths": test_paths,
        }

    def classify_image(self, image_path: str | Path) -> str:
        prediction = self.predict([image_path])
        return str(prediction[0])


def load_split(split_dir: str | Path) -> tuple[list[Path], list[str]]:
    classifier = BagOfVisualWordsClassifier()
    return classifier._iter_image_files(split_dir)


def train_and_evaluate(
    dataset_root: str | Path,
    feature_extractor: str = "SIFT",
    vocabulary_size: int = 128,
    max_descriptors_per_image: int = 250,
    max_vocabulary_descriptors: int = 20000,
    random_state: int = 42,
):
    dataset_root = Path(dataset_root)
    config = BoVWConfig(
        feature_extractor=feature_extractor,
        vocabulary_size=vocabulary_size,
        max_descriptors_per_image=max_descriptors_per_image,
        max_vocabulary_descriptors=max_vocabulary_descriptors,
        random_state=random_state,
    )
    model = BagOfVisualWordsClassifier(config)
    model.fit(dataset_root / "train")
    metrics = model.evaluate(dataset_root / "test")
    return model, metrics
