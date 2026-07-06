#!/bin/bash

# Download the InsightFace model pack used for face detection and recognition.
# The pack is extracted into ./models/<pack>/ where the recognition backend
# (see ../recognition.py) expects to find the .onnx files.

# The pack to download, override by passing a name as the first argument
PACK="${1:-buffalo_s}"

BASE_URL="https://github.com/deepinsight/insightface/releases/download/v0.7"
ARCHIVE="${PACK}.zip"
TARGET="models/${PACK}"

echo "Downloading InsightFace model pack '${PACK}'..."

# Prefer wget, fall back on curl
if hash wget 2>/dev/null; then
	# Check if wget supports the option to only show the progress bar
	wget --help | grep -q "\--show-progress" && \
		_PROGRESS_OPT="-q --show-progress" || _PROGRESS_OPT=""

	wget $_PROGRESS_OPT --tries 5 "${BASE_URL}/${ARCHIVE}"
else
	curl --location --retry 5 --output "${ARCHIVE}" "${BASE_URL}/${ARCHIVE}"
fi

echo " "
echo "Unpacking into ${TARGET}..."

# The archive contains the .onnx files, extract them (flat) into the pack directory
mkdir -p "${TARGET}"
unzip -o -j "${ARCHIVE}" -d "${TARGET}"

# Remove the downloaded archive
rm -f "${ARCHIVE}"

echo "Done."
