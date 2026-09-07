"""Unit tests for the OpenCV/NumPy feature-extraction module.

These are lower-level than the API tests: they check that each engineered
feature actually responds in the expected direction to a known synthetic
degradation, which is the real evidence that the "computer vision
understanding" behind the feature set is sound (rather than just checking
that the code runs without raising).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.image_features import (  # noqa: E402
    FEATURE_NAMES,
    ImageDecodeError,
    decode_image_bytes,
    extract_features,
    validate_upload,
)


def _synthetic_image(seed=0, size=256) -> np.ndarray:
    """A reasonably sharp, well-exposed synthetic BGR test image with
    some structure (not flat) so sharpness/edge features are meaningful."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    for _ in range(15):
        pt1 = tuple(rng.integers(0, size, 2).tolist())
        pt2 = tuple(rng.integers(0, size, 2).tolist())
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, pt1, pt2, color, 2)
    return img


class TestFeatureExtractionBasics(unittest.TestCase):
    def test_returns_all_16_named_features(self):
        img = _synthetic_image()
        feats = extract_features(img).to_dict()
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), 16)

    def test_feature_vector_matches_declared_order(self):
        img = _synthetic_image()
        feats = extract_features(img)
        vector = feats.to_vector()
        as_dict = feats.to_dict()
        for name, value in zip(FEATURE_NAMES, vector):
            self.assertAlmostEqual(float(value), as_dict[name], places=5)


class TestFeatureDirectionality(unittest.TestCase):
    """Each of these checks that a specific, controlled degradation moves
    the corresponding feature in the theoretically expected direction."""

    def setUp(self):
        self.base = _synthetic_image()

    def test_blur_lowers_laplacian_variance(self):
        sharp = extract_features(self.base).laplacian_variance
        blurred_img = cv2.GaussianBlur(self.base, (21, 21), 0)
        blurred = extract_features(blurred_img).laplacian_variance
        self.assertLess(blurred, sharp)

    def test_darkening_increases_dark_pixel_ratio(self):
        normal = extract_features(self.base).dark_pixel_ratio
        darkened_img = np.clip(self.base.astype(np.int16) * 0.2, 0, 255).astype(np.uint8)
        darkened = extract_features(darkened_img).dark_pixel_ratio
        self.assertGreater(darkened, normal)

    def test_brightening_increases_bright_pixel_ratio(self):
        normal = extract_features(self.base).bright_pixel_ratio
        brightened_img = np.clip(self.base.astype(np.int16) + 150, 0, 255).astype(np.uint8)
        brightened = extract_features(brightened_img).bright_pixel_ratio
        self.assertGreater(brightened, normal)

    def test_gaussian_noise_increases_noise_estimate(self):
        clean = extract_features(self.base).noise_estimate
        rng = np.random.default_rng(1)
        noisy_img = np.clip(
            self.base.astype(np.float32) + rng.normal(0, 40, self.base.shape), 0, 255
        ).astype(np.uint8)
        noisy = extract_features(noisy_img).noise_estimate
        self.assertGreater(noisy, clean)

    def test_heavy_jpeg_compression_increases_blockiness(self):
        baseline = extract_features(self.base).blockiness
        ok, enc = cv2.imencode(".jpg", self.base, [cv2.IMWRITE_JPEG_QUALITY, 3])
        self.assertTrue(ok)
        compressed_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        compressed = extract_features(compressed_img).blockiness
        self.assertGreater(compressed, baseline)

    def test_flat_image_has_low_entropy(self):
        flat = np.full((128, 128, 3), 100, dtype=np.uint8)
        textured = self.base
        self.assertLess(extract_features(flat).entropy, extract_features(textured).entropy)


class TestValidation(unittest.TestCase):
    def test_rejects_empty_file(self):
        with self.assertRaises(ImageDecodeError):
            validate_upload("photo.jpg", "image/jpeg", b"")

    def test_rejects_unsupported_extension(self):
        with self.assertRaises(ImageDecodeError):
            validate_upload("photo.gif", "image/gif", b"12345")

    def test_rejects_oversized_file(self):
        big = b"\x00" * (11 * 1024 * 1024)
        with self.assertRaises(ImageDecodeError):
            validate_upload("photo.jpg", "image/jpeg", big)

    def test_decode_raises_on_garbage_bytes(self):
        with self.assertRaises(ImageDecodeError):
            decode_image_bytes(b"not an image at all")

    def test_decode_succeeds_on_real_jpeg(self):
        ok, enc = cv2.imencode(".jpg", _synthetic_image())
        self.assertTrue(ok)
        img = decode_image_bytes(enc.tobytes())
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
