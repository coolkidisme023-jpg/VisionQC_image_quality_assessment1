# VisionQC frontend image.
#
# This frontend is plain HTML/CSS/JS (ES modules) with no build step (see
# SUBMISSION_NOTES.md for why this replaced React + Vite), so the image
# simply serves the static files through nginx - no node/npm stage needed.
#
# Build (from the frontend/ directory):
#     docker build -t visionqc-frontend .

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html
COPY env-config.template.js /usr/share/nginx/html/env-config.template.js
COPY env-config.js /usr/share/nginx/html/env-config.js
COPY src/ /usr/share/nginx/html/src/
COPY docker-entrypoint.sh /docker-entrypoint.sh

RUN chmod +x /docker-entrypoint.sh

# Overridden at `docker run`/compose time to point at the backend's
# externally-reachable URL (the browser calls this directly, so it must
# be reachable from the user's machine, not just inside the Docker network).
ENV VITE_API_URL=http://localhost:8000

EXPOSE 5173

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
