#!/usr/bin/env bash

# Register a moving image to a reference using FSL FLIRT (6 DOF).
# Usage:
#   ./scripts/register_mton_to_mton.sh <moving_nii.gz> <reference_nii.gz> [output_dir]
# If output_dir is omitted, outputs are written beside the moving image.
# Outputs (generic names):
#   - <moving_basename>_to_<ref_basename>_flirt6dof.nii.gz (resampled image)
#   - <moving_basename>_to_<ref_basename>_flirt6dof.mat (affine transform matrix)

set -euo pipefail

if ! command -v flirt >/dev/null 2>&1; then
  echo "FSL FLIRT (flirt) is not on PATH; please source the FSL setup (e.g., . /usr/local/fsl/etc/fslconf/fsl.sh)" 1>&2
  exit 1
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <moving_nii.gz> <reference_nii.gz> [output_dir]" 1>&2
  exit 1
fi

moving=$1
ref=$2
out_dir=${3:-"$(dirname "$moving")"}

if [[ ! -d "$out_dir" ]]; then
  echo "Output directory does not exist: $out_dir" 1>&2
  exit 1
fi

moving_base=$(basename "$moving" .nii.gz)
ref_base=$(basename "$ref" .nii.gz)
prefix="r2_"

out="${out_dir%/}/${prefix}${moving_base}_to_${ref_base}_flirt6dof.nii.gz"
omat="${out_dir%/}/${prefix}${moving_base}_to_${ref_base}_flirt6dof.mat"

for f in "$ref" "$moving"; do
  if [[ ! -f "$f" ]]; then
    echo "Required file not found: $f" 1>&2
    exit 1
  fi
done

echo "Running FLIRT 6-DOF registration:"
echo "  Moving image : $moving"
echo "  Reference    : $ref"
echo "  Output image : $out"
echo "  Matrix       : $omat"

flirt \
  -in "$moving" \
  -ref "$ref" \
  -out "$out" \
  -omat "$omat" \
  -dof 6 \
  -cost mutualinfo \
  -searchrx -90 90 \
  -searchry -90 90 \
  -searchrz -90 90 \
  -interp spline

echo "Done."
echo "Resliced image : $out"
echo "Transform mat  : $omat"
