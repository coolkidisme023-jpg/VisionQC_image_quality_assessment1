#!/bin/sh
# Regenerate env-config.js from the VITE_API_URL environment variable
# every time the frontend container starts, then hand off to nginx.
set -e

: "${VITE_API_URL:=http://localhost:8000}"

sed "s|__VITE_API_URL__|${VITE_API_URL}|g" \
    /usr/share/nginx/html/env-config.template.js > /usr/share/nginx/html/env-config.js

echo "VisionQC frontend: API base URL = ${VITE_API_URL}"

exec "$@"
