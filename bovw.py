from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import pairwise_distances

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
        self._reset_rng()

    def _reset_rng(self) -> None:
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
        self._reset_rng()
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

    def index_dataset(self, image_paths: Sequence[str | Path], use_tfidf: bool = False) -> None:
        """
        Build an index of BoVW histograms for the given image paths.

        Args:
            image_paths: Sequence of image file paths to index.
            use_tfidf: If True, fit a TF-IDF transformer on the histograms and
                      store TF-IDF-weighted features for retrieval.
        """
        if self.kmeans is None:
            raise RuntimeError("Call fit() before indexing dataset.")
        # Ensure deterministic sampling
        self._reset_rng()
        # Compute histograms for all images
        paths = [Path(p) for p in image_paths]
        features = self.transform(paths)
        self._index_paths = paths
        self._tfidf = None
        if use_tfidf:
            transformer = TfidfTransformer(norm="l2", use_idf=True)
            features = transformer.fit_transform(features).toarray()
            self._tfidf = transformer
        # store as float32
        self._index_features = np.asarray(features, dtype=np.float32)
        self._index_use_tfidf = use_tfidf

    def query(self, query_image: str | Path, top_k: int = 5, metric: str = "cosine", use_tfidf: bool | None = None):
        """
        Retrieve top-k images similar to the query image.

        Args:
            query_image: Path to the query image.
            top_k: Number of neighbors to return.
            metric: Similarity metric to use: 'cosine' or 'euclidean'.
            use_tfidf: If True/False, override whether TF-IDF is applied to the
                       query. If None, uses whatever was used when indexing.

        Returns:
            List of tuples (Path, score) where score is similarity for cosine
            or distance for euclidean. Sorted best-first.
        """
        if not hasattr(self, "_index_features") or len(self._index_features) == 0:
            raise RuntimeError("Index is empty. Call index_dataset() before querying.")
        q_desc = self.extract_descriptors(query_image)
        q_hist = self._image_histogram(q_desc).reshape(1, -1)
        # Apply TF-IDF if requested or if index used it
        index_uses_tfidf = bool(getattr(self, "_index_use_tfidf", False))
        if use_tfidf is None:
            use_tfidf = index_uses_tfidf
        if use_tfidf != index_uses_tfidf:
            raise RuntimeError("TF-IDF usage must match the way the index was built. Rebuild or reload the index with the desired setting.")
        if use_tfidf:
            if getattr(self, "_tfidf", None) is None:
                raise RuntimeError("Index was not built with TF-IDF. Rebuild index with use_tfidf=True to use TF-IDF.")
            q_hist = self._tfidf.transform(q_hist).toarray()

        feats = self._index_features
        if metric.lower() == "cosine":
            sims = cosine_similarity(q_hist, feats)[0]
            # higher is better
            idx = np.argsort(-sims)[:top_k]
            return [(self._index_paths[i], float(sims[i])) for i in idx]
        elif metric.lower() in {"euclidean", "l2"}:
            dists = np.linalg.norm(feats - q_hist, axis=1)
            idx = np.argsort(dists)[:top_k]
            return [(self._index_paths[i], float(dists[i])) for i in idx]
        else:
            # fallback to sklearn pairwise distances for other metrics
            dists = pairwise_distances(q_hist, feats, metric=metric)[0]
            idx = np.argsort(dists)[:top_k]
            return [(self._index_paths[i], float(dists[i])) for i in idx]

    def save_index(self, index_path: str | Path) -> Path:
        if not hasattr(self, "_index_features") or len(self._index_features) == 0:
            raise RuntimeError("Build an index before saving it.")
        index_path = Path(index_path)
        if index_path.suffix == "":
            index_path = index_path.with_suffix(".pkl")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_paths": [str(path) for path in self._index_paths],
            "index_features": self._index_features,
            "index_use_tfidf": bool(getattr(self, "_index_use_tfidf", False)),
            "tfidf": getattr(self, "_tfidf", None),
        }
        with index_path.open("wb") as file_handle:
            pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        return index_path

    def load_index(self, index_path: str | Path) -> "BagOfVisualWordsClassifier":
        index_path = Path(index_path)
        with index_path.open("rb") as file_handle:
            payload = pickle.load(file_handle)
        self._index_paths = [Path(path) for path in payload["index_paths"]]
        self._index_features = np.asarray(payload["index_features"], dtype=np.float32)
        self._index_use_tfidf = bool(payload.get("index_use_tfidf", False))
        self._tfidf = payload.get("tfidf", None)
        return self

    def evaluate_retrieval(
        self,
        query_paths: Sequence[str | Path],
        top_k: int = 10,
        metric: str = "cosine",
        use_tfidf: bool | None = None,
        exclude_query: bool = True,
    ) -> dict:
        """
        Evaluate retrieval performance for a set of queries.

        Returns a dict with per-query APs and aggregated Precision@k, Recall@k, and mAP.
        """
        if not hasattr(self, "_index_features") or len(self._index_features) == 0:
            raise RuntimeError("Index is empty. Call index_dataset() before evaluating retrieval.")
        # Use full ranking for AP calculation (full index)
        index_size = len(self._index_paths)
        per_query_results = []
        precisions = []
        recalls = []
        aps = []
        for qp in query_paths:
            qp = Path(qp)
            # get full ranking
            ranked = self.query(qp, top_k=index_size, metric=metric, use_tfidf=use_tfidf)
            ranked_paths = [p for p, _ in ranked]
            # Optionally remove the query itself from the ranking
            if exclude_query:
                ranked_paths = [p for p in ranked_paths if p != qp]
            # compute relevance vector (True if same class)
            query_class = qp.parent.name
            relevant = [1 if p.parent.name == query_class else 0 for p in ranked_paths]
            total_relevant = sum(1 for p in self._index_paths if p.parent.name == query_class and (not exclude_query or p != qp))
            # Precision@k and Recall@k
            p_at_k = precision_at_k(relevant, top_k)
            r_at_k = recall_at_k(relevant, top_k, total_relevant)
            ap = average_precision(relevant)
            per_query_results.append({"query": str(qp), "precision_at_k": p_at_k, "recall_at_k": r_at_k, "ap": ap})
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            aps.append(ap)

        mean_p_at_k = float(np.mean(precisions)) if precisions else 0.0
        mean_r_at_k = float(np.mean(recalls)) if recalls else 0.0
        mean_ap = float(np.mean(aps)) if aps else 0.0
        return {
            "per_query": per_query_results,
            "mean_precision_at_k": mean_p_at_k,
            "mean_recall_at_k": mean_r_at_k,
            "mean_average_precision": mean_ap,
            "per_query_ap": aps,
        }

    def save(self, model_path: str | Path) -> Path:
        if self.kmeans is None or self.classifier is None:
            raise RuntimeError("Train the model before saving it.")
        model_path = Path(model_path)
        if model_path.suffix == "":
            model_path = model_path.with_suffix(".pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "kmeans": self.kmeans,
            "classifier": self.classifier,
            "class_names": self.class_names,
        }
        with model_path.open("wb") as file_handle:
            pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model_path

    @classmethod
    def load(cls, model_path: str | Path) -> "BagOfVisualWordsClassifier":
        model_path = Path(model_path)
        with model_path.open("rb") as file_handle:
            payload = pickle.load(file_handle)
        model = cls(payload["config"])
        model.kmeans = payload["kmeans"]
        model.classifier = payload["classifier"]
        model.class_names = list(payload["class_names"])
        return model


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


def precision_at_k(relevant: Sequence[int | bool], k: int) -> float:
    """Compute Precision@k for a binary relevance list (1/0 or True/False).

    Args:
        relevant: sequence indicating relevance in ranked order (1/0 or True/False).
        k: cutoff for precision.
    Returns:
        Precision@k as float.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    rel = np.asarray(relevant, dtype=np.int32)
    top = rel[:k]
    return float(np.sum(top) / k)


def recall_at_k(relevant: Sequence[int | bool], k: int, total_relevant: int | None = None) -> float:
    """Compute Recall@k.

    Args:
        relevant: sequence indicating relevance in ranked order.
        k: cutoff for recall.
        total_relevant: total number of relevant items in the collection. If None,
                        uses the number of relevant items present in `relevant`.
    Returns:
        Recall@k as float.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    rel = np.asarray(relevant, dtype=np.int32)
    if total_relevant is None:
        total_relevant = int(np.sum(rel))
    if total_relevant == 0:
        return 0.0
    top_hits = int(np.sum(rel[:k]))
    return float(top_hits / total_relevant)


def average_precision(relevant: Sequence[int | bool]) -> float:
    """Compute Average Precision (AP) for a binary relevance sequence.

    AP = sum_k (Precision@k * rel_k) / num_relevant
    Returns 0.0 if there are no relevant documents.
    """
    rel = np.asarray(relevant, dtype=np.int32)
    num_relevant = int(np.sum(rel))
    if num_relevant == 0:
        return 0.0
    precisions = []
    for i in range(1, len(rel) + 1):
        if rel[i - 1]:
            precisions.append(np.sum(rel[:i]) / float(i))
    if not precisions:
        return 0.0
    return float(np.sum(precisions) / num_relevant)


def mean_average_precision(list_of_relevant: Sequence[Sequence[int | bool]]) -> float:
    """Compute Mean Average Precision (mAP) over a list of relevance sequences.

    Each element in `list_of_relevant` is a ranked binary relevance sequence for a query.
    """
    aps = [average_precision(seq) for seq in list_of_relevant]
    if not aps:
        return 0.0
    return float(np.mean(aps))
