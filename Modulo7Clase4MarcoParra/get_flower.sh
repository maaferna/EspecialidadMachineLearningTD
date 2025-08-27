# get_flowers.sh
#!/usr/bin/env bash
set -e

DATA_DIR="data/flower_photos"
TGZ="flower_photos.tgz"
URL="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

mkdir -p data
cd data

if [ ! -d "flower_photos" ]; then
  echo "Descargando flower_photos..."
  wget -q "${URL}" -O "${TGZ}"
  echo "Descomprimiendo..."
  tar -xzf "${TGZ}"
  rm -f "${TGZ}"
else
  echo "flower_photos ya existe. Omitiendo descarga."
fi

echo "✓ Dataset en $(pwd)/flower_photos"
