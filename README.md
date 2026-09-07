# VisionQC — AI-Powered Image Quality & Defect Detection

VisionQC is a full-stack application that accepts an uploaded image, runs it
through a local computer-vision + machine-learning pipeline, and reports an
interpretable 0–100 quality score, a predicted condition, a list of detected
issues (with severity/confidence/explanation), and full explainability —
all **without calling any external AI/LLM API**. Every model runs locally;
no API key is required or used anywhere in this project.

> **Read this first:** this project was developed in a network-sandboxed
> environment (no `pip`/`npm`/`apt` access) and substitutes Flask + SQLite
> (`sqlite3`) + vanilla JS for the originally-suggested FastAPI + SQLAlchemy
> + React/Vite stack, so that every line of code could actually be run and
> verified rather than shipped untested. **See [`SUBMISSION_NOTES.md`](SUBMISSION_NOTES.md)
> for the full rationale and a checklist of what was verified.**

---

## 1. Project overview

Upload a photo → VisionQC decodes and validates it → extracts 16 engineered
OpenCV/NumPy image-quality features → feeds them into a trained
`RandomForestClassifier` → combines the model's output with the raw
statistics into a documented quality score and a structured issue list →
persists the result to SQLite → returns structured JSON → renders it on a
live dashboard.

## 2. Features

- Drag-and-drop (or click-to-browse) image upload with client-side preview
- Detects: **blur**, **underexposure**, **overexposure**, **noise**,
  **corruption/severe degradation**, and **localized visual defects**
  (scratches, dead-pixel blocks, dust, color blotches)
- 0–100 quality score with a 5-tier label (Excellent → Severely Degraded)
- Per-issue severity, confidence, human-readable explanation, and the
  underlying statistic that triggered it
- Full explainability: raw statistics, class probabilities, and the Random
  Forest's top global feature importances
- Analysis history with pagination, detail view, and delete
- Robust corruption handling (invalid file, unsupported format, unreadable/
  truncated image, oversized upload) with structured JSON errors and
  correct HTTP status codes
- Responsive dashboard (desktop/tablet/mobile), with loading/error/success
  states throughout
- 28 automated tests, all passing; a fully reproducible local training
  pipeline with real (not fabricated) evaluation metrics

## 3. Architecture

```
IMAGE UPLOAD (multipart/form-data)
        │
        ▼
FILE VALIDATION        (extension, size ≤10MB, non-empty)
        │
        ▼
IMAGE DECODING          (Pillow verify + decode → OpenCV BGR array)
        │
        ▼
IMAGE PREPROCESSING     (resize to a working resolution)
        │
        ▼
OPENCV/NUMPY FEATURE EXTRACTION   (16 engineered features)
        │
        ▼
TRAINED ML MODEL         (StandardScaler + RandomForestClassifier)
        │
        ▼
CLASS PROBABILITIES      (7-way softmax-like distribution)
        │
        ▼
ISSUE DETECTION           (hybrid: statistic threshold + model probability)
        │
        ▼
QUALITY SCORE              (documented weighted formula, 0-100)
        │
        ▼
EXPLAINABILITY               (stats + probabilities + feature importances)
        │
        ▼
DATABASE STORAGE               (SQLite, via app/database.py)
        │
        ▼
REST API RESPONSE (JSON)         (Flask, app/routes/*)
        │
        ▼
DASHBOARD (vanilla JS SPA)        (frontend/)
```

**Layers**

| Layer | Location | Responsibility |
|---|---|---|
| Feature extraction | `backend/app/ml/image_features.py` | Pure OpenCV/NumPy, no framework dependency |
| Training pipeline | `backend/training/` | Synthetic dataset generation + model training/eval |
| Inference service | `backend/app/services/analyzer.py` | Model loading, scoring, issue detection, explainability |
| Persistence | `backend/app/database.py` | SQLite repository (`AnalysisRecord` CRUD) |
| API | `backend/app/routes/`, `backend/app/main.py` | Flask routes, CORS, error handling |
| Frontend | `frontend/` | Vanilla ES-module dashboard, no build step |

## 4. Technology stack

| Concern | Used | Originally suggested | Why changed |
|---|---|---|---|
| Backend framework | **Flask 3.1** | FastAPI | No network access to `pip install fastapi` in the dev sandbox (see notes) |
| ORM/DB | **SQLite via `sqlite3`** (stdlib) | SQLAlchemy | Same reason; a thin repository class (`app/database.py`) keeps the same clean API |
| Frontend | **Vanilla HTML/CSS/JS (ES modules)** | React + Vite | No `npm install` access; explicitly allowed by the assessment brief ("plain HTML/CSS/JavaScript... may be used") |
| CV | OpenCV, NumPy, Pillow, scikit-image | — | as planned |
| ML | scikit-learn `RandomForestClassifier`, joblib | — | as planned |
| DB | SQLite | — | as planned |
| Tests | `unittest` (pytest-compatible) | pytest | pytest itself wasn't installed in the sandbox; tests are pytest-discoverable |
| Deployment | Docker, Docker Compose, Nginx | — | as planned |

No external AI/LLM API (OpenAI, Claude, Gemini, etc.) is called anywhere in
this codebase, and no API key is required to run it.

## 5. ML approach

A `RandomForestClassifier` (scikit-learn) is trained on the 16 engineered
features to classify an image into one of 7 classes: `clean`, `blur`,
`underexposed`, `overexposed`, `noise`, `corrupted`, `defect`. Random Forest
was chosen because:

- it handles the nonlinear, non-monotonic relationships between raw
  statistics and quality classes well (e.g. brightness is "bad" at both
  extremes, not linearly),
- it is fast at inference time (a hard requirement for an interactive
  upload-and-analyze UX),
- it provides built-in, model-native feature importances for
  explainability, and
- with only 16 tabular features and a few hundred training samples, it is
  far less prone to overfitting than a deep network, and needs no GPU.

The model is a 2-step `sklearn.pipeline.Pipeline`: `StandardScaler` →
`RandomForestClassifier(n_estimators=400, max_depth=18, min_samples_leaf=2,
class_weight="balanced", random_state=42, n_jobs=-1)`.

Class probabilities from this model are **one input** to the final quality
score, not the whole answer — see §9/§10 and `compute_quality_score()` in
`analyzer.py`.

## 6. Feature engineering

All 16 features are computed in `backend/app/ml/image_features.py`
(`extract_features()`), purely from OpenCV/NumPy — no learned feature
extractor:

| # | Feature | What it measures |
|---|---|---|
| 1 | `mean_brightness` | Average pixel intensity — under/over-exposure |
| 2 | `brightness_std` | Spread of intensity — flatness/contrast |
| 3 | `dark_pixel_ratio` | Fraction of very dark pixels — underexposure |
| 4 | `bright_pixel_ratio` | Fraction of near-white pixels — overexposure/blown highlights |
| 5 | `rms_contrast` | Normalized intensity std — haze/washed-out look |
| 6 | `laplacian_variance` | Variance of the 2nd derivative — the classic blur/sharpness indicator |
| 7 | `mean_gradient_magnitude` | Average Sobel gradient — fine detail presence |
| 8 | `edge_density` | Fraction of Canny-detected edge pixels — structural detail |
| 9 | `noise_estimate` | Residual energy after median denoising — sensor/compression noise |
| 10 | `entropy` | Shannon entropy of the intensity histogram — information content |
| 11 | `mean_saturation` | Average HSV saturation — washed-out/near-grayscale detection |
| 12 | `saturation_std` | Spread of saturation — reinforces #11 |
| 13 | `colorfulness` | Hasler–Süsstrunk colorfulness metric |
| 14 | `blockiness` | 8×8 JPEG block-boundary gradient ratio — compression damage |
| 15 | `texture_energy` | Local variance of a high-pass filtered image — flat/degraded regions |
| 16 | `dynamic_range` | 99th–1st percentile intensity spread — tonal quality |

Every feature's description is also served live at
`/api/analyze` → `explainability.feature_descriptions`.

## 7. Dataset generation

No external dataset or download is used. `backend/training/degradations.py`
synthesizes labeled training examples from **19 natural photographs and
scientific images bundled locally with scikit-image** (`skimage.data` —
`astronaut`, `coffee`, `chelsea`, `rocket`, `camera`, `coins`, `retina`,
etc.; no network fetch required). For each source image, controlled, seeded
degradations are applied to produce examples of all 7 classes:

- **clean** — mild gain/bias jitter + high-quality JPEG re-encode
- **blur** — Gaussian or motion blur, random kernel size/angle
- **underexposed** — multiplicative gain + gamma darkening
- **overexposed** — multiplicative gain + additive brightening
- **noise** — additive Gaussian noise or salt-and-pepper noise
- **corrupted** — heavy JPEG re-compression (quality 2–12) + random block
  overwrite + extreme downsample/upsample
- **defect** — 3–6 localized synthetic scratches, dead-pixel blocks, dust
  spots, or color blotches drawn onto an otherwise clean image

One source image, `checkerboard`, was deliberately excluded after it was
found (during development — see the confusion-matrix investigation in the
commit history of this file) to be a perfectly periodic synthetic pattern
that aliases with the 8×8 `blockiness` feature and produced pathological
outlier values unrepresentative of real photographs.

Each source image is also resized to one of several target resolutions
(320–576px) rather than a single fixed size, so the "clean" class spans a
realistic range of scales instead of overfitting to one working resolution.

19 sources × 7 classes × 6 variants = **798 total samples**.

## 8. Training procedure

Run: `python backend/training/train_model.py`

1. Load the 19 local source images.
2. Generate 6 seeded degraded variants per class per source (deterministic
   — `random_state=42`).
3. Extract all 16 features for every sample → build `X` (798×16), `y`
   (798 labels), and `groups` (798 source-image ids).
4. **Grouped train/test split** via `sklearn.model_selection.GroupShuffleSplit`
   (`test_size=0.30`, `random_state=42`) — split on the **source-image
   group**, not individual samples, so no two degraded versions of the same
   original photo can appear on both sides of the split. This is asserted
   in code (`assert not overlap`) and printed at training time.
5. Fit `StandardScaler → RandomForestClassifier` on the training split only.
6. Evaluate on the held-out test split (see §9).
7. Save `models/quality_model.joblib`, `models/model_metadata.json`,
   `models/metrics.json`, `models/confusion_matrix.png`,
   `models/feature_importance.png`, and one representative **test-set**
   sample image per class to `models/samples/`.

## 9. Evaluation

These are the **actual** numbers produced by `train_model.py` on the
held-out, grouped test split (13 train source images / 6 test source
images, 546 train / 252 test samples) — not manufactured. Full detail in
[`models/metrics.json`](models/metrics.json).

| Metric | Value |
|---|---|
| Accuracy | **0.889** |
| Precision (macro) | 0.906 |
| Recall (macro) | 0.889 |
| F1-score (macro) | 0.888 |
| ROC-AUC (macro, one-vs-rest) | **0.986** |

Per-class:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| blur | 0.919 | 0.944 | 0.932 |
| clean | 1.000 | 0.583 | 0.737 |
| corrupted | 0.711 | 0.889 | 0.790 |
| defect | 0.714 | 0.833 | 0.769 |
| noise | 1.000 | 1.000 | 1.000 |
| overexposed | 1.000 | 0.972 | 0.986 |
| underexposed | 1.000 | 1.000 | 1.000 |

See `models/confusion_matrix.png` and `models/feature_importance.png`.

**Discussion / failure cases.** `blur`, `noise`, `overexposed`, and
`underexposed` are detected almost perfectly — they have strong, global,
unambiguous statistical signatures. `clean`, `corrupted`, and `defect` are
the hardest three-way boundary: `clean`'s recall (0.58) is the weakest spot
— roughly 40% of true "clean" test images are called `corrupted` or
`defect`. Inspecting the confusion matrix, this is a real and explainable
limitation, not a bug: a handful of clean test photos (e.g. `astronaut`,
`chelsea`, `rocket`, `cat` — busy, colorful, high-detail images) sit close
to the `defect`/`corrupted` decision boundary in feature space, because
those classes are also, by construction, derived from otherwise-sharp
images with added local artifacts — global statistics alone can't always
tell "this image is naturally rich in color/edges" from "this image has a
color blotch/heavy compression added to it." This is expected: `defect` in
particular is a *localized* phenomenon, and this system deliberately uses
*global* image statistics (see §18, Limitations). The quality-score formula
(§10) partially compensates for this: even when the top-1 class is wrong,
the score still blends in directly-measured sharpness/exposure/noise
statistics, so a genuinely clean image still tends to land in the
"Acceptable"/"Good" range rather than being scored as if it were actually
degraded.

## 10. Explainability

Every analysis response includes an `explainability` block with:

- `predicted_class` and `confidence`
- the full `class_probabilities` distribution over all 7 classes
- the model's `top_feature_importances` (Random Forest Gini importance —
  real, model-native, not a post-hoc approximation), each with a
  plain-language description
- `feature_descriptions` for all 16 features
- a `note` stating plainly that this is a classical-ML model with **no**
  Grad-CAM/saliency map (the project does not claim spatial heatmap
  localization it doesn't actually produce)

In addition, every entry in `issues[]` carries its own `explanation` and
the exact `statistic` (name + value) that triggered it, so a person can
verify the reasoning without touching the model at all.

## 11. API documentation

Interactive reference: `GET /docs` (self-contained HTML, no CDN
dependency). Machine-readable spec: `GET /openapi.json`.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness/readiness check |
| POST | `/api/analyze` | Upload + analyze an image (`multipart/form-data`, field `image`) |
| GET | `/api/analyses?limit=&offset=` | Paginated history, newest first |
| GET | `/api/analyses/{id}` | Full stored analysis |
| DELETE | `/api/analyses/{id}` | Delete a stored analysis |

Example:

```bash
curl -F "image=@photo.jpg" http://localhost:8000/api/analyze
```

```json
{
  "id": 1,
  "filename": "photo.jpg",
  "predicted_class": "blur",
  "quality_score": 61.5,
  "quality_label": "DEGRADED",
  "issues": [
    {
      "type": "blur",
      "severity": "high",
      "confidence": 0.91,
      "explanation": "Low Laplacian variance indicates insufficient image sharpness...",
      "statistic": {"name": "laplacian_variance", "value": 18.4}
    }
  ],
  "probabilities": {"blur": 0.7, "clean": 0.05, "...": 0.0},
  "statistics": {"mean_brightness": 90.1, "...": 0.0},
  "explainability": {"top_feature_importances": ["..."]},
  "model_version": "1.0.0",
  "created_at": "2026-01-01T00:00:00+00:00"
}
```

```bash
curl http://localhost:8000/api/analyses?limit=10
curl http://localhost:8000/api/analyses/1
curl -X DELETE http://localhost:8000/api/analyses/1
```

## 12. Local installation

```bash
git clone <this repo> && cd image_quality_assessment
cp .env.example .env   # optional, defaults work out of the box
```

Requires Python 3.11+. No Node/npm installation is required for the
frontend (plain static files).

## 13. Backend startup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Train the model (writes models/quality_model.joblib + evaluation artifacts)
python training/train_model.py

# Run the API (development server)
python -m flask --app app.main run --host 0.0.0.0 --port 8000
# or, for a production-style server:
gunicorn --bind 0.0.0.0:8000 --workers 2 app.main:app
```

Visit `http://localhost:8000/health` to confirm it's running, and
`http://localhost:8000/docs` for API docs.

## 14. Frontend startup

```bash
cd frontend
python3 -m http.server 5173
# or: npx serve .  /  any static file server
```

Open `http://localhost:5173`. By default it talks to the backend at
`http://localhost:8000` (see `env-config.js`); edit that file to point
elsewhere if needed.

## 15. Docker startup

```bash
docker compose up --build
```

- Backend: `http://localhost:8000` (health: `/health`, docs: `/docs`)
- Frontend: `http://localhost:5173`

The SQLite database is bind-mounted at `./data/visionqc.db` so it persists
across container restarts/rebuilds. The model is copied into the backend
image at build time and also bind-mounted read-write from `./models`, so
retraining on the host (`python backend/training/train_model.py`) and
restarting the backend container picks up the new model without a rebuild.

> Docker Compose/Dockerfiles were authored and manually path-verified but
> **could not be build-tested** in the development sandbox (no Docker
> daemon available there — see `SUBMISSION_NOTES.md`). `docker compose
> config` was used to validate the compose file's syntax.

## 16. Testing

```bash
cd backend
python -m unittest discover -s tests -v   # works anywhere, stdlib only
# or, once pytest is installed (e.g. via requirements.txt):
pytest -q
```

28 tests, all passing (verified in this repository's development
environment): health endpoint, valid/invalid/unsupported/oversized/corrupted
uploads, analysis persistence, history pagination, deletion, docs
endpoints, and feature-extraction directionality checks (e.g. "does
Gaussian blur actually lower `laplacian_variance`?").

## 17. Project structure

```
image_quality_assessment/
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   │   └── image_features.py      # 16-feature extraction (OpenCV/NumPy)
│   │   ├── services/
│   │   │   └── analyzer.py            # scoring, issue detection, explainability
│   │   ├── routes/
│   │   │   ├── health.py, analyses.py, docs.py
│   │   ├── database.py                # SQLite repository
│   │   ├── config.py                  # env-var driven settings
│   │   └── main.py                    # Flask app factory
│   ├── training/
│   │   ├── degradations.py            # synthetic degradation generators
│   │   └── train_model.py             # full training/eval pipeline
│   ├── tests/
│   │   ├── test_api.py, test_features.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/                # scoreGauge, issueCard, statsPanel, resultView
│   │   ├── pages/                     # analyzePage, historyPage
│   │   ├── services/api.js
│   │   ├── app.js, styles.css
│   │   ├── index.html
│   ├── nginx.conf, Dockerfile, docker-entrypoint.sh
├── models/
│   ├── quality_model.joblib, model_metadata.json, metrics.json
│   ├── confusion_matrix.png, feature_importance.png
│   └── samples/                       # one representative test-set image per class
├── docker-compose.yml
├── .env.example
├── README.md
└── SUBMISSION_NOTES.md
```

## 18. Limitations

- **Synthetic training data.** The model is trained entirely on synthetic
  degradations of 19 stock photos, not a labeled real-world dataset.
  Real-world performance on arbitrary user photos will likely differ from
  the held-out synthetic evaluation numbers above — this has **not** been
  validated against real-world labeled data, and this project makes no
  claim of production-grade accuracy.
- **`defect` detection is global, not localized.** The `defect` class is
  detected from whole-image statistics, not spatial localization. A small
  scratch in the corner of a large photo may not shift global statistics
  enough to be detected; conversely, a naturally busy/colorful photo can
  resemble the synthetic "defect" feature signature (see §9 discussion).
  There is no bounding box or heatmap showing *where* a defect is.
- **No Grad-CAM/saliency map.** This is a classical ML model (Random
  Forest over engineered features), not a CNN — there is no spatial
  gradient to visualize, and the app does not claim otherwise.
- **`clean` vs. `corrupted`/`defect` boundary is the weakest.** See the
  per-class evaluation discussion in §9.
- **Single-image inference latency**, not optimized for high-throughput
  batch workloads.
- **Offline development sandbox.** See `SUBMISSION_NOTES.md` for the
  FastAPI→Flask / SQLAlchemy→sqlite3 / React+Vite→vanilla-JS substitutions
  and what could/couldn't be verified as a result.

## 19. Future improvements

- Collect or license a real labeled image-quality dataset (or run a manual
  labeling pass on real photos) to validate/retrain beyond synthetic data.
- Add a lightweight CNN (transfer learning, e.g. MobileNet) as an
  alternative model for a Grad-CAM-style spatial heatmap, particularly to
  localize `defect` regions.
- Patch-based / sliding-window feature extraction for spatial defect
  localization within classical ML (no deep learning required).
- Confidence calibration (e.g. Platt scaling / isotonic regression) on the
  Random Forest's probabilities.
- Batch analysis endpoint for multiple images in one request.
- Model versioning with a registry of multiple trained models and A/B
  comparison.
- CI pipeline running `pytest` + a frontend smoke test on every push.

## 20. Demo instructions

1. `cd backend && pip install -r requirements.txt && python training/train_model.py`
2. `python -m flask --app app.main run --port 8000` (backend)
3. In a second terminal: `cd frontend && python3 -m http.server 5173`
4. Open `http://localhost:5173`, drag in a photo, click **Analyze image**.
5. Check the **History** tab to see it persisted, click a row to view its
   full detail again.
6. Try an invalid file (e.g. a `.txt` renamed to `.jpg`, or a file over
   10 MB) to see the error states.

Or, with Docker: `docker compose up --build`, then open
`http://localhost:5173`.
