"""
Synthetic image-quality degradation generators.

Because the assessment prohibits calling any external AI/vision API, and no
pre-labeled "image quality" dataset is bundled with the project, VisionQC
builds its own local, fully reproducible training set: it starts from a
handful of clean natural images (bundled with scikit-image, no network
access required) and applies controlled, seeded degradations to synthesize
labeled examples of every target class.

Each ``make_*`` function takes a clean RGB uint8 image and a NumPy
``Generator`` (already seeded upstream) and returns a new RGB uint8 image
plus a short human-readable description of the degradation applied (useful
for the sample gallery / debugging).

Classes produced:
    clean          - mild, realistic variation of the original (no quality
                     defect introduced)
    blur           - Gaussian or motion blur (insufficient sharpness)
    underexposed   - darkened via gain/gamma (too dark)
    overexposed    - brightened via gain/gamma (too bright, blown highlights)
    noise          - additive Gaussian or salt-and-pepper sensor noise
    corrupted      - severe degradation: heavy JPEG re-compression combined
                     with block-level corruption (unreadable / garbage data)
    defect         - localized synthetic visual defects: scratches, dead
                     pixel blocks, sensor dust, color-cast blotches
"""

from __future__ import annotations

import cv2
import numpy as np

CLASSES = ["clean", "blur", "underexposed", "overexposed", "noise", "corrupted", "defect"]


def _clip_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def make_clean(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild, realistic variation: slight brightness/contrast jitter and a
    high-quality JPEG re-encode (lossy but not visually degraded)."""
    out = img.astype(np.float32)
    gain = rng.uniform(0.98, 1.02)
    bias = rng.uniform(-3, 3)
    out = out * gain + bias
    out = _clip_u8(out)
    quality = int(rng.integers(93, 100))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return out


def make_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian or motion blur simulating focus/handshake blur."""
    if rng.random() < 0.6:
        k = int(rng.choice([9, 13, 17, 21, 25]))
        out = cv2.GaussianBlur(img, (k, k), 0)
    else:
        k = int(rng.choice([9, 15, 21]))
        kernel = np.zeros((k, k), dtype=np.float32)
        angle = rng.uniform(0, 180)
        kernel[k // 2, :] = 1.0
        m = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, m, (k, k))
        kernel /= kernel.sum() + 1e-8
        out = cv2.filter2D(img, -1, kernel)
    return _clip_u8(out)


def make_underexposed(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken the image via multiplicative gain + gamma (too dark)."""
    gain = rng.uniform(0.12, 0.42)
    gamma = rng.uniform(1.4, 2.6)
    out = (img.astype(np.float32) / 255.0) ** gamma
    out = out * 255.0 * gain / max(gain, 0.2)
    out = out * gain
    return _clip_u8(out)


def make_overexposed(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brighten the image via additive/multiplicative gain (blown highlights)."""
    gain = rng.uniform(1.6, 2.6)
    bias = rng.uniform(40, 100)
    out = img.astype(np.float32) * gain + bias
    return _clip_u8(out)


def make_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise or salt-and-pepper noise."""
    out = img.astype(np.float32)
    if rng.random() < 0.65:
        sigma = rng.uniform(25, 50)
        noise = rng.normal(0, sigma, img.shape)
        out = out + noise
        out = _clip_u8(out)
    else:
        amount = rng.uniform(0.02, 0.08)
        out = img.copy()
        h, w = img.shape[:2]
        n_salt = int(amount * h * w * 0.5)
        n_pepper = int(amount * h * w * 0.5)
        ys = rng.integers(0, h, n_salt)
        xs = rng.integers(0, w, n_salt)
        out[ys, xs] = 255
        ys = rng.integers(0, h, n_pepper)
        xs = rng.integers(0, w, n_pepper)
        out[ys, xs] = 0
    return _clip_u8(out)


def make_corrupted(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Severe degradation: heavy re-compression plus block-level corruption,
    simulating a badly damaged / partially unreadable file."""
    out = img.copy()
    h, w = out.shape[:2]

    # Heavy JPEG re-compression (very low quality -> strong blockiness).
    quality = int(rng.integers(2, 12))
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    # Random block corruption: overwrite several blocks with garbage/solid
    # values to mimic a partially unreadable capture.
    n_blocks = int(rng.integers(4, 12))
    for _ in range(n_blocks):
        bw = int(rng.integers(w // 12, w // 4))
        bh = int(rng.integers(h // 12, h // 4))
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        if rng.random() < 0.5:
            out[y0:y0 + bh, x0:x0 + bw] = rng.integers(0, 255, (bh, bw, 3), dtype=np.uint8)
        else:
            out[y0:y0 + bh, x0:x0 + bw] = int(rng.integers(0, 255))

    # Extreme downsample/upsample to destroy fine structure further.
    scale = rng.uniform(0.08, 0.2)
    small = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return _clip_u8(out)


def make_defect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Localized synthetic visual defects on an otherwise well-exposed,
    sharp image: scratches, dead-pixel blocks, dust spots, color blotches."""
    out = img.copy()
    h, w = out.shape[:2]
    n_defects = int(rng.integers(3, 7))

    for _ in range(n_defects):
        kind = rng.choice(["scratch", "dead_block", "dust", "color_blotch"])
        if kind == "scratch":
            pt1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            pt2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            thickness = int(rng.integers(1, 4))
            cv2.line(out, pt1, pt2, color, thickness)
        elif kind == "dead_block":
            bw, bh = int(rng.integers(8, 30)), int(rng.integers(8, 30))
            x0 = int(rng.integers(0, max(1, w - bw)))
            y0 = int(rng.integers(0, max(1, h - bh)))
            color = int(rng.choice([0, 255]))
            out[y0:y0 + bh, x0:x0 + bw] = color
        elif kind == "dust":
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(4, 16))
            color = tuple(int(c) for c in rng.integers(0, 60, 3))
            cv2.circle(out, (cx, cy), r, color, -1)
        else:  # color_blotch
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(20, 60))
            overlay = out.copy()
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.circle(overlay, (cx, cy), r, color, -1)
            alpha = rng.uniform(0.3, 0.6)
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    return _clip_u8(out)


DEGRADATION_FUNCS = {
    "clean": make_clean,
    "blur": make_blur,
    "underexposed": make_underexposed,
    "overexposed": make_overexposed,
    "noise": make_noise,
    "corrupted": make_corrupted,
    "defect": make_defect,
}


def generate_variant(label: str, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dispatch to the degradation generator for ``label``."""
    if label not in DEGRADATION_FUNCS:
        raise ValueError(f"Unknown class label: {label}")
    return DEGRADATION_FUNCS[label](img, rng)
