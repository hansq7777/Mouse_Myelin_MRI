#!/bin/bash
# Script for converting Enhanced MR dicoms to nifti
# First argument: enhanced dicom filename (including .dcm)
#

if [ "$#" -ne 1 ]; then
  echo "Need exactly 1 inputs: enhanced dicom filename"
  exit 1
fi

# Support passing either a single DICOM file or a directory containing one DICOM.
SRC="${1}"
if [ -d "$SRC" ]; then
  mapfile -d '' DCM_FILES < <(find "$SRC" -maxdepth 1 -type f -name '*.dcm' -print0 | LC_ALL=C sort -z)
  if [ ${#DCM_FILES[@]} -eq 0 ]; then
    echo "No .dcm files found in directory: $SRC" >&2
    exit 1
  fi
  if [ ${#DCM_FILES[@]} -gt 1 ]; then
    echo "Warning: multiple .dcm files in $SRC; using first: ${DCM_FILES[0]}" >&2
  fi
  SRC="${DCM_FILES[0]}"
fi

if [ ! -f "$SRC" ]; then
  echo "DICOM not found: $SRC" >&2
  exit 1
fi

INPUT=$(readlink -f "$SRC")

BASEDIR=$(dirname "$0")
DIR=$(dirname "${INPUT}")
FILE=$(basename "${INPUT}")
FILE="${FILE%.*}"

# Get path of this script, and assume brukHead.py resides there too
SCRIPTPATH=$(readlink -f "$0")
BPATH=$(dirname "$SCRIPTPATH")

# Find a dcm2niix binary: prefer colocated copy, then Windows path, then PATH
if [ -x "$BPATH/dcm2niix" ]; then
  DCM2NIIX_BIN="$BPATH/dcm2niix"
elif [ -x "$BPATH/dcm2niix.exe" ]; then
  DCM2NIIX_BIN="$BPATH/dcm2niix.exe"
elif [ -x "/mnt/c/tools/dcm2niix/dcm2niix.exe" ]; then
  DCM2NIIX_BIN="/mnt/c/tools/dcm2niix/dcm2niix.exe"
else
  DCM2NIIX_BIN=$(command -v dcm2niix || true)
fi

if [ -z "$DCM2NIIX_BIN" ]; then
  echo "dcm2niix not found (looked in $BPATH, /mnt/c/tools/dcm2niix, and PATH)" >&2
  exit 1
fi

# If using Windows binary, convert paths to Windows format
INPUT_FOR_DCM2NIIX="${INPUT}"
DIR_FOR_DCM2NIIX="${DIR}"
if [[ "$DCM2NIIX_BIN" == *.exe ]]; then
  if command -v wslpath >/dev/null 2>&1; then
    INPUT_FOR_DCM2NIIX=$(wslpath -w "${INPUT}")
    DIR_FOR_DCM2NIIX=$(wslpath -w "${DIR}")
  fi
fi

"$DCM2NIIX_BIN" -z y -o "${DIR_FOR_DCM2NIIX}" -f "${FILE}" "${INPUT_FOR_DCM2NIIX}"

# Extract header info
PYTHON_BIN=${PYTHON_BIN:-$(command -v python || true)}
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(command -v python3 || true)
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "python not found (checked python, python3); set PYTHON_BIN to override" >&2
  exit 1
fi

"$PYTHON_BIN" "$BPATH/brukHead.py" "${INPUT}"
