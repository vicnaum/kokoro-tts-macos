#!/bin/bash
# Downloads the Kokoro model + voice embeddings into Resources/.
# Both files come from the KokoroSwift author's test app repo, so they are
# guaranteed compatible with the library version this project pins.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p Resources

echo "Downloading voice embeddings (voices.npz, ~15 MB)..."
curl -L --fail --progress-bar \
  -o Resources/voices.npz \
  "https://github.com/mlalma/KokoroTestApp/raw/main/Resources/voices.npz"

echo "Downloading Kokoro model (kokoro-v1_0.safetensors, ~600 MB, Git LFS)..."
curl -L --fail --progress-bar \
  -o Resources/kokoro-v1_0.safetensors \
  "https://github.com/mlalma/KokoroTestApp/raw/main/Resources/kokoro-v1_0.safetensors"

echo
echo "Verifying..."
ls -lh Resources/
SIZE=$(stat -f%z Resources/kokoro-v1_0.safetensors)
if [ "$SIZE" -lt 100000000 ]; then
  echo "WARNING: kokoro-v1_0.safetensors is suspiciously small ($SIZE bytes)."
  echo "The Git LFS redirect may have failed — download it manually from:"
  echo "https://github.com/mlalma/KokoroTestApp/blob/main/Resources/kokoro-v1_0.safetensors"
  exit 1
fi
echo "Done. Now run: xcodegen && open KokoroVoice.xcodeproj"
