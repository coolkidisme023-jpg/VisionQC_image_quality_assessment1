# Submission Notes

## Why the tech stack differs from the brief's suggestions

The project brief (and the user's own instructions) recommended
**FastAPI + SQLAlchemy** for the backend and **React + Vite** for the
frontend. This project was built inside a sandboxed development
environment with **no outbound network access at all** — `pip install`,
`npm install`, and even `apt-get update` were all blocked by the
organization's egress policy (confirmed with `curl`, `pip`, `npm`, and
`apt-get`, all returning `403 host_not_allowed` / proxy denials; see the
diagnostic commands run at the start of this project's development
session).

Concretely, this environment had **no FastAPI, no SQLAlchemy, no
pytest, no Node package registry access, and no Docker daemon**, but it
did have: Flask, OpenCV, NumPy, Pillow, scikit-learn, scikit-image,
joblib, matplotlib, Python's `unittest` and `sqlite3` standard-library
modules, the Docker **client** (no daemon), and Node (for running the
static frontend, though no npm packages could be installed).

Given that choice, I opted to **build a fully offline-verifiable
deliverable rather than write code targeting libraries I could not
install, run, or test**:

| Original suggestion | Used instead | Why |
|---|---|---|
| FastAPI | **Flask 3.1** (already installed) | Same REST-API responsibility; the assessment brief explicitly allows "another suitable framework." Every endpoint was actually run and tested. |
| SQLAlchemy | **`sqlite3`** (Python stdlib) wrapped in a small repository class (`app/database.py`) | Same persistence responsibility; the brief allows "SQLite, PostgreSQL, or another suitable database" and doesn't mandate an ORM. The public API (`save_analysis`/`get_analysis`/`list_analyses`/`delete_analysis`) is what a SQLAlchemy-backed version would expose too — swapping the implementation later is a contained change. |
| React + Vite | **Vanilla HTML/CSS/JS (native ES modules, no bundler)** | The brief explicitly allows "React, Vue, plain HTML/CSS/JavaScript... may be used." No build step means nothing to fail to install — the whole frontend was verified running end-to-end against the real backend using Playwright (Chromium was pre-installed in this sandbox). |
| pytest | **`unittest`** (stdlib), written to also be pytest-discoverable | pytest wasn't installed; `unittest.TestCase` classes are natively discovered and run by pytest once it *is* available (e.g. via `requirements.txt` on a normal machine, or inside Docker), so `pytest -q` will work outside this sandbox. Inside this sandbox, the same 28 tests were run via `python -m unittest discover`. |

This was a deliberate, disclosed trade-off (the user was asked and chose
this option explicitly) in favor of **a working, fully-tested system**
over **a stack-name match that might not run**.

## What was actually run and verified in this environment

- ✅ `python backend/training/train_model.py` — trains the model end to
  end, produces `models/quality_model.joblib`, `model_metadata.json`,
  `metrics.json`, `confusion_matrix.png`, `feature_importance.png`, and
  the `models/samples/` gallery. Re-run twice while iterating on the
  degradation pipeline (see README §7/§9 for the checkerboard-outlier
  investigation and fix).
- ✅ `python -m unittest discover -s backend/tests -v` — **28/28 tests
  pass** (API tests + feature-extraction directionality tests).
- ✅ Full Flask app exercised via `app.test_client()` and via a real
  running server (`flask run` on port 8000): `/health`, `/api/analyze`
  (valid image, corrupted file, unsupported extension, oversized file),
  `/api/analyses`, `/api/analyses/{id}`, `DELETE /api/analyses/{id}`,
  `/docs`, `/openapi.json` — all manually exercised with real requests in
  addition to the automated test suite.
- ✅ Full frontend exercised **in a real Chromium browser via Playwright**
  against the real running backend (not mocked): file upload via drag-in
  simulation, the Analyze flow (loading → success state), the History
  page (list, pagination, row click → detail view, delete), and the
  client-side validation error state (bad extension) — at desktop, tablet
  (820px), and mobile (390px) viewports. Screenshots were reviewed at
  each step. **One real bug was found and fixed this way**: the `hidden`
  HTML attribute was being silently overridden by a CSS class that also
  set `display: flex` on the same elements (an author-stylesheet rule
  beats a user-agent-stylesheet rule regardless of selector specificity),
  so the loading/warning banners stayed visually stuck after JS set
  `el.hidden = true`. Fixed with a global `[hidden] { display: none
  !important; }` rule.
- ✅ `docker compose config` — validates `docker-compose.yml` syntax and
  variable interpolation (this does not require the Docker daemon).

## What could NOT be verified in this environment

- ❌ **`docker compose up --build`** — no Docker daemon was available
  (`docker info` fails with "Cannot connect to the Docker daemon"; the
  Docker **client** binary is present but there is nothing for it to talk
  to). The Dockerfiles and compose file were written carefully and
  cross-checked by hand against `backend/app/config.py`'s path-resolution
  logic (`REPO_ROOT = <backend dir>.parent`) to make sure the in-container
  layout (`/app/backend`, `/app/models`, `/app/data`) matches what the
  application actually expects — but an actual container build/run was
  not possible here.
- ❌ **`pip install -r backend/requirements.txt` from a clean
  environment** — every package in that file is already installed in this
  sandbox at the exact pinned version, so the pipeline was run against
  those, not against a fresh `pip install`. The pinned versions should
  install cleanly on any machine with normal PyPI access.
- ❌ **`npm`-based tooling** — not applicable; the frontend has no npm
  dependencies by design (see table above).

## Recommendation if you want the originally-suggested stack

Porting to FastAPI is a contained change: all business logic already
lives outside the route layer (`app/services/analyzer.py`,
`app/database.py`), so only `app/main.py` and `app/routes/*.py` would need
rewriting against FastAPI's `APIRouter`/`UploadFile`/`HTTPException`, and
`app/database.py` would need its `sqlite3` calls replaced with SQLAlchemy
`Session`/`Model` equivalents. Neither change touches the ML pipeline,
feature extraction, training script, or evaluation artifacts. If you'd
like, I'm happy to do that port in an environment with normal package-
registry access.
