/**
 * Runtime configuration for the VisionQC frontend.
 *
 * Loaded as a plain <script> tag BEFORE the application modules, so it can
 * set global config the app reads at startup - this is the equivalent of
 * a Vite `VITE_API_URL` env var, but resolved at container/page load time
 * instead of at a JS build step (this frontend has no build step; see
 * SUBMISSION_NOTES.md for why).
 *
 * Local/dev default: backend running on http://localhost:8000 (as
 * requested in the project spec, frontend on 5173 / backend on 8000).
 *
 * In Docker, docker-entrypoint.sh regenerates this exact file from the
 * VITE_API_URL environment variable when the frontend container starts,
 * so docker-compose.yml's `VITE_API_URL` setting is what actually takes
 * effect in a container deployment.
 */
window.__VISIONQC_API_BASE__ = "http://localhost:8000";
