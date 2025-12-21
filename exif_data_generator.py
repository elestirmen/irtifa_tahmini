import os
import numpy as np
from typing import List, Sequence, Tuple
from PIL import Image
import tensorflow as tf


class ExifSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that reads images from filepaths and extracts altitude labels
    directly from EXIF metadata (GPSInfo/GPSAltitude) on the fly.

    - Avoids a precomputed CSV index
    - Optionally caches labels in-memory to reduce EXIF re-reads across epochs
    - Filters out files without a valid altitude when cache_labels=True
    """

    def __init__(
        self,
        filepaths: Sequence[str],
        target_size: Tuple[int, int] = (512, 512),
        batch_size: int = 16,
        color_mode: str = 'rgb',  # 'rgb' or 'grayscale'
        rescale: float = 1.0 / 255.0,
        shuffle: bool = True,
        cache_labels: bool = True,
    ) -> None:
        self.filepaths: List[str] = list(filepaths)
        self.target_size = target_size
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.channels = 3 if color_mode == 'rgb' else 1
        self.rescale = rescale
        self.shuffle = shuffle
        self.cache_labels = cache_labels

        self.indexes = np.arange(len(self.filepaths))
        self.label_cache: dict[str, float] = {}

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.cache_labels:
            self._precache_labels_and_filter_invalids()

    def __len__(self) -> int:
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _read_image(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if self.color_mode == 'grayscale':
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img = img.resize(self.target_size)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        if self.rescale:
            arr *= self.rescale
        return arr

    def _extract_altitude_from_exif(self, path: str) -> float:
        # Use cache if available
        if self.cache_labels and path in self.label_cache:
            return self.label_cache[path]

        altitude = np.nan
        try:
            img = Image.open(path)
            exif = getattr(img, '_getexif', lambda: None)()
            if exif and 34853 in exif:  # GPSInfo tag
                gps = exif[34853]
                alt = gps.get(6, None)       # GPSAltitude
                alt_ref = gps.get(5, 0)      # GPSAltitudeRef (0 = above sea level, 1 = below)
                if alt is not None:
                    # Handle PIL's Rational or tuple-like (num, den)
                    num = getattr(alt, 'numerator', None)
                    den = getattr(alt, 'denominator', None)
                    if num is not None and den not in (None, 0):
                        altitude = float(num) / float(den)
                    elif isinstance(alt, (tuple, list)) and len(alt) == 2 and alt[1] != 0:
                        altitude = float(alt[0]) / float(alt[1])
                    else:
                        altitude = float(alt)

                    if alt_ref == 1:  # below sea level
                        altitude = -abs(altitude)
        except Exception:
            altitude = np.nan

        if self.cache_labels:
            self.label_cache[path] = altitude
        return altitude

    def _precache_labels_and_filter_invalids(self) -> None:
        valid_idx: List[int] = []
        for i, p in enumerate(self.filepaths):
            alt = self._extract_altitude_from_exif(p)
            if not np.isnan(alt):
                valid_idx.append(i)

        if len(valid_idx) != len(self.filepaths):
            # Filter out images without valid labels
            self.filepaths = [self.filepaths[i] for i in valid_idx]
            self.indexes = np.arange(len(self.filepaths))
            if self.shuffle:
                np.random.shuffle(self.indexes)

    def __getitem__(self, idx: int):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = [self.filepaths[i] for i in batch_idx]
        images = []
        labels = []
        for path in batch_files:
            images.append(self._read_image(path))
            labels.append(self._extract_altitude_from_exif(path))
        x = np.stack(images, axis=0)
        y = np.array(labels, dtype=np.float32)
        return x, y

