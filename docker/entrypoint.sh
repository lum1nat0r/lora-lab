#!/bin/sh
set -e

# The config directory is usually a bind mount and starts out empty. Seed it
# from the examples baked into the image, without ever overwriting an existing
# file. If the mount is read-only the app still works: load_project falls back
# to /app/defaults for any missing top-level file.
mkdir -p /app/config
if [ ! -f /app/config/project.yaml ]; then
  cp -n /app/defaults/*.yaml /app/config/ 2>/dev/null || true
  echo "lora-lab: seeded /app/config from image defaults"
fi

exec "$@"
