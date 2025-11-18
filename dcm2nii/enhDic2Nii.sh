#!/bin/bash
# Script for converting Enhanced MR dicoms to nifti
# First argument: enhanced dicom filename (including .dcm)
#

if [ "$#" -ne 1 ]; then
  echo "Need exactly 1 inputs: enhanced dicom filename"
  exit 1
fi

BASEDIR=$(dirname $0)
DIR=$(dirname "${1}")
FILE=$(basename "${1}")
FILE="${FILE%.*}"

# Get path of this script, and assume brukHead.py resides there too
SCRIPTPATH=$(readlink -f "$0")
BPATH=$(dirname "$SCRIPTPATH")

dcm2niix -z y -o ${DIR} -f ${FILE} ${1}

# Extract header info
python $BPATH/brukHead.py ${1} 

