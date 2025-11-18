#!/bin/bash
# Script to preprocess dMRI data - includes PCA denoising (can be turned on or off), Gibbs ringing correction, susceptibility and eddy current induced distortion correction
# Need 2 to 6 inputs: preprocName nifti_AP nifti_PA dodenoise refInd skipToStep. Do not include extensions (e.g., .nii.gz).
# preprocName: name of output preprocessed NIFTI file
# nifti_AP: name of NIFTI file with phase encoding in anterior-posterior (AP) direction
# nifti_PA: name of NIFTI file with phase encoding in opposite direction from nifti_AP (so posterior-anterior)
# To not have a nifti_PA input, can either not enter or set to NONE
# dodenoise: 0 (no denoising) or 1 (denoising); default is 0
# Use refInd to use eddy_correct instead of eddy. Necessary for when there is no b0

# In this script FSL is being run from a singularity container and the data is accessed from a mounted directory. The script may be modified as needed. 

shopt -s expand_aliases
source ~/.bash_aliases

S="singularity exec --nv /mnt/baron1_local1/containers/singularity-fsl_6.0.3.sif"

# Read in input
if [ "$#" -gt 6 ]; then
  echo "Need 2 to 6 inputs: preprocName nifti_AP nifti_PA dodenoise refInd skipToStep. Do not include extensions (e.g., .nii.gz)."
  exit 1
fi
if [ "$#" -lt 2 ]; then
  echo "Need 2 to 6 inputs: preprocName nifti_AP nifti_PA dodenoise refInd skipToStep. Do not include extensions (e.g., .nii.gz)."
  exit 1
fi
# if preprocName already exists (prevents overwriting data)
if test -f "$1.nii.gz"; then
    echo "error: $1.nii.gz exists"
    exit 1
fi

im_preproc=$1
im_AP=$2
if [ "$#" -gt 2 ]; then
    im_PA=$3
else
    im_PA=NONE
fi
if [ "$#" -gt 3 ]; then
    DODENOISE=$4

else
    DODENOISE=0
fi
if [ "$#" -gt 4 ]; then
    REFIND=$5
else
    REFIND=-1
fi
if [ "$#" -gt 5 ]; then
    TOSTEP=$6
else
    TOSTEP=-1
fi

ERRFLAG=0

SCRIPTPATH=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPTPATH")

#copy everything to the home dir, since FSL in a singularity container seems to have trouble reading stuff from a mounted dir
cp $im_AP.nii.gz aveComb.nii.gz 
cp ${im_AP}_preproc_mask.nii.gz aveComb_mask.nii.gz
cp $im_AP.bval aveComb.bval
cp $im_AP.bvec aveComb.bvec

NVOL=$(fslnvols aveComb)
echo $NVOL

# Merge all data for better denoising (step 1)
if [ $TOSTEP -lt 2 ]; then
    if [ "$im_PA" != "NONE" ]; then
	cp $im_PA.nii.gz aveComb_PA.nii.gz    
        fslmerge -t data_all aveComb.nii.gz aveComb_PA.nii.gz
	NVOL_PA=$(fslnvols aveComb_PA)
    else
        cp aveComb.nii.gz data_all.nii.gz
	
    fi

    # Denoise
    if [ $DODENOISE -eq 1 ]; then
        dwidenoise -force data_all.nii.gz tmp1.nii.gz
    else	
        cp data_all.nii.gz tmp1.nii.gz
    fi
fi

# Gibbs correction (step 2)
if [ $TOSTEP -lt 3 ]; then
    mrdegibbs -force tmp1.nii.gz data_all.nii.gz
    rm tmp1.nii.gz
fi

# Readout time
RTIME=0.1

# Find indices for lowest b-value shell
BINDS=$(python "$SCRIPTPATH"/getb0Inds.py "$im_AP".bval "$im_AP".bvec)        # b0 indices
echo $BINDS
BINDSA=($BINDS)
REFVOL=${BINDSA[0]}
if [ $REFIND -ge 0 ]; then
    REFVOL=$REFIND
fi

echo "before topup"
# Do topup operations (step 3) 
if [ $TOSTEP -lt 4 ]; then
    if [ "$im_PA" != "NONE" ]; then
        # Create nifti of lowest b-shell images and tmp_acqparams.txt file required for topup
        COUNT=0
	# If nifti_AP and nifti_PA have the same number of volumes
        if [ $NVOL -eq $NVOL_PA ]
        then
            echo "NVOL and NVOL_PA are equal"
            for i in $BINDS
            do
                fslroi data_all tmp1 $i 1
                fslroi data_all tmp2 $(($i+$NVOL)) 1
                if [ $COUNT -eq 0 ]
                then
                    fslmerge -t tmp3 tmp1 tmp2
                    printf "0 1 0 $RTIME\n0 -1 0 $RTIME\n" > tmp_acqparams.txt
                else
                    fslmerge -t tmp3 tmp3 tmp1 tmp2
                    printf "0 1 0 $RTIME\n0 -1 0 $RTIME\n" >> tmp_acqparams.txt
                fi
                ((COUNT=COUNT+1))
            done
        else
            echo "NVOL and NVOL_PA are not equal"
            for i in $BINDS
            do
                fslroi data_all tmp1 $i 1
                printf "0 1 0 $RTIME\n" >> tmp_acqparams.txt
                echo $i
                if [ $COUNT -eq 0 ]
                then
                    fslmerge -t tmp3 tmp1
                else
                    fslmerge -t tmp3 tmp3 tmp1
                fi
                ((COUNT=COUNT+1))
            done
            for ((i=1;i<=NVOL_PA;i++)); 
            do
                fslroi data_all tmp2 $(($i+$NVOL-1)) 1
                printf "0 -1 0 $RTIME\n" >> tmp_acqparams.txt
                echo $i
                fslmerge -t tmp3 tmp3 tmp2
            done
        fi


        # Estimate field with topup 
        echo "running topup"
	topup --imain=tmp3 --datain=tmp_acqparams.txt --out=tmp_topup_results --fout=offresfield --config=b02b0.cnf --fwhm=0.8 --warpres=1 -v

        if [ $? -ne 0 ]; then
            exit 1
        fi

        # Separate back out for applying topup in eddy_cuda
        fslroi data_all tmp1 0 $NVOL
        fslroi data_all tmp2 $NVOL -1

        fslmaths data_all -abs data_all

    # if topup is not run, create tmp_acqparams.txt file required for eddy
    else
        printf "0 1 0 0.1\n" > tmp_acqparams.txt
    fi

    # OGSE may have no b0's acquired, which eddy does not like. So, create an artificial one
    if [ 1 -eq 0 ]; then
        # TODO: this would be better than eddy_correct, since it will correct for interslice motion
        fslroi data_all tmp1 $REFIND 1
        fslmaths tmp1 -mul 2 tmp2
        fslmerge -t data_all tmp2
        ((NVOL=NVOL+1))
    fi

    # Generate AP/PA direction info. 
    indx=""
    for ((i=1; i<=$NVOL; i+=1)); do indx="$indx 1"; echo $i; done
    echo $indx > tmp_index.txt
    echo tmp_index.txt
fi

# Run eddy (step 4)
if [ $TOSTEP -lt 5 ]; then
    if [ $REFIND -ge 0 ]; then
        echo "running fsl eddy_correct"
        eddy_correct data_all preproc $REFIND
        if [ $? -ne 0 ]; then
            exit 1
        fi
    else
        echo "running fsl eddy"
	# Run eddy_cuda with topup applied
	if [ "$im_PA" != "NONE" ]; then
            eddy_cuda --imain=tmp1 --mask=aveComb_mask --acqp=tmp_acqparams.txt --index=tmp_index.txt --bvecs=aveComb.bvec --bvals=aveComb.bval --topup=tmp_topup_results --out=preproc -v --data_is_shelled
        # Run eddy_cuda without topup	
        else
            eddy_cuda --imain=data_all --mask=aveComb_mask --acqp=tmp_acqparams.txt --index=tmp_index.txt --bvecs=aveComb.bvec --bvals=aveComb.bval --out=preproc -v --data_is_shelled
	fi
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
fi

fslmaths preproc -abs preproc

# Copy over bvals and bvecs
cp preproc.eddy_rotated_bvecs $im_preproc.bvec 
cp $im_AP.bval $im_preproc.bval

# Copy niftis from home dir to the correct location
cp preproc.nii.gz $im_preproc.nii.gz
cp aveComb_mask.nii.gz ${im_preproc}_mask.nii.gz

dir="$(dirname "${im_preproc}")"
cp preproc.eddy* $dir

# Cleanup extra files
rm *.nii.gz
rm tmp*
