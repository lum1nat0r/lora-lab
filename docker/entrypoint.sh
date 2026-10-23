#!/bin/sh
set -e

# The config directory is usually a bind mount and may start empty (or
# half-filled after a manual fix). Seed every missing top-level file from the
# examples baked into the image, never overwriting an existing one.
mkdir -p /app/config
seeded=""
for file in /app/defaults/*.yaml; do
  [ -e "$file" ] || continue
  name=$(basename "$file")
  if [ ! -f "/app/config/$name" ]; then
    cp "$file" "/app/config/$name" 2>/dev/null && seeded="$seeded $name" || true
  fi
done
if [ -n "$seeded" ]; then
  echo "lora-lab: seeded /app/config from image defaults:$seeded"
fi

exec "$@"
