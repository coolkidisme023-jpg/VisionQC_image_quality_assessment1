"""
API test suite for VisionQC.

Written as ``unittest.TestCase`` subclasses so the suite is runnable in
two ways:
    1. ``python -m unittest discover -s backend/tests`` (stdlib only,
       works in any environment - this is how these tests were actually
       executed and verified during development, since this offline
       sandbox does not have pytest installed).
    2. ``pytest -q`` (pytest auto-discovers and runs unittest.TestCase
       classes natively, so this file satisfies "tests can run using
       pytest -q" once pytest is available, e.g. inside the Docker image
       or on a machine with normal internet access).

Uses Flask's built-in ``app.test_client()`` (the Flask equivalent of
FastAPI's ``TestClient``) so no real HTTP server or network socket is
required.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.database as database_module  # noqa: E402
from app.config import settings  # noqa: E402


def _make_image_bytes(width=200, height=200, color=(120, 130, 140), fmt="JPEG") -> bytes:
    """Build an in-memory synthetic test image (no fixture files needed)."""
    rng = np.random.default_rng(0)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    # add a bit of structure/edges so it isn't perfectly flat
    arr[height // 4: height // 2, width // 4: width // 2] = (200, 60, 60)
    noise = rng.integers(-4, 4, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class VisionQCApiTestCase(unittest.TestCase):
    """Base class: gives every test an isolated, temporary SQLite database
    so tests never touch the real development database and never leak
    state into one another."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        test_db_path = Path(self._tmpdir.name) / "test_visionqc.db"

        # Point the app at an isolated database for this test and force
        # the process-wide DB singleton to be rebuilt against it.
        settings.DATABASE_PATH = test_db_path
        database_module._db_instance = None

        from app.main import create_app  # imported here so config changes above apply
        self.app = create_app()
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmpdir.cleanup()
        database_module._db_instance = None


class TestHealthEndpoint(VisionQCApiTestCase):
    def test_health_returns_200_and_status_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("model_file_found", body)
        self.assertIn("version", body)


class TestAnalyzeEndpoint(VisionQCApiTestCase):
    def test_valid_image_upload_returns_full_analysis(self):
        image_bytes = _make_image_bytes()
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(image_bytes), "sample.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.get_json()

        for key in ("id", "filename", "predicted_class", "quality_score",
                    "quality_label", "issues", "probabilities", "statistics",
                    "explainability", "model_version", "created_at"):
            self.assertIn(key, body, f"missing key '{key}' in response")

        self.assertIsInstance(body["id"], int)
        self.assertGreaterEqual(body["quality_score"], 0)
        self.assertLessEqual(body["quality_score"], 100)
        self.assertIn(body["quality_label"], {
            "EXCELLENT", "GOOD", "ACCEPTABLE", "DEGRADED", "SEVERELY_DEGRADED",
        })
        self.assertIsInstance(body["issues"], list)
        self.assertAlmostEqual(sum(body["probabilities"].values()), 1.0, places=2)
        self.assertIn("top_feature_importances", body["explainability"])

    def test_missing_file_field_returns_400(self):
        resp = self.client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_invalid_unreadable_file_returns_400(self):
        garbage = b"this is not an image file, just plain bytes" * 10
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(garbage), "photo.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_unsupported_format_returns_400(self):
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(b"hello world"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file extension", resp.get_json()["error"])

    def test_oversized_upload_returns_413(self):
        oversized = b"\x00" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(oversized), "huge.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 413)
        self.assertIn("error", resp.get_json())

    def test_corrupted_image_bytes_with_valid_extension_returns_400(self):
        truncated_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # JPEG magic bytes, then garbage
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(truncated_jpeg), "corrupt.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)


class TestAnalysisPersistenceAndHistory(VisionQCApiTestCase):
    def _upload(self, filename="sample.jpg"):
        image_bytes = _make_image_bytes()
        resp = self.client.post(
            "/api/analyze",
            data={"image": (io.BytesIO(image_bytes), filename)},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 201)
        return resp.get_json()

    def test_analysis_is_persisted_and_retrievable(self):
        created = self._upload()
        resp = self.client.get(f"/api/analyses/{created['id']}")
        self.assertEqual(resp.status_code, 200)
        fetched = resp.get_json()
        self.assertEqual(fetched["id"], created["id"])
        self.assertEqual(fetched["filename"], "sample.jpg")
        self.assertEqual(fetched["quality_score"], created["quality_score"])

    def test_get_nonexistent_analysis_returns_404(self):
        resp = self.client.get("/api/analyses/999999")
        self.assertEqual(resp.status_code, 404)

    def test_history_endpoint_lists_created_analyses(self):
        self._upload("first.jpg")
        self._upload("second.jpg")

        resp = self.client.get("/api/analyses")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["total"], 2)
        self.assertEqual(len(body["items"]), 2)
        filenames = {item["filename"] for item in body["items"]}
        self.assertEqual(filenames, {"first.jpg", "second.jpg"})

    def test_history_pagination_limit(self):
        for i in range(5):
            self._upload(f"img{i}.jpg")
        resp = self.client.get("/api/analyses?limit=2&offset=0")
        body = resp.get_json()
        self.assertEqual(body["total"], 5)
        self.assertEqual(len(body["items"]), 2)

    def test_delete_analysis_removes_it(self):
        created = self._upload()
        del_resp = self.client.delete(f"/api/analyses/{created['id']}")
        self.assertEqual(del_resp.status_code, 204)

        get_resp = self.client.get(f"/api/analyses/{created['id']}")
        self.assertEqual(get_resp.status_code, 404)

    def test_delete_nonexistent_analysis_returns_404(self):
        resp = self.client.delete("/api/analyses/999999")
        self.assertEqual(resp.status_code, 404)


class TestDocsEndpoints(VisionQCApiTestCase):
    def test_docs_page_available(self):
        resp = self.client.get("/docs")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"VisionQC API", resp.data)

    def test_openapi_json_available(self):
        resp = self.client.get("/openapi.json")
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertIn("/api/analyze", body["paths"])


if __name__ == "__main__":
    unittest.main()
