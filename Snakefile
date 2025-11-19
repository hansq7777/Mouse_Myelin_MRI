# Snakemake Pipeline for mouse data from 9.4 T - MT MRI only (MTR/MTsat)
# Reduced to MT-related rules.
import numpy as np
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.resolve()
MATLAB_DIR = REPO_ROOT / "mt_pipeline" / "matlab"
DCM2NII_DIR = REPO_ROOT / "mt_pipeline" / "dcm2nii"

# Convert anatomical/MT DICOMs to NIFTIs
rule dcmTOnii_anat:
    input:
        dicom = "{filepath}/{name}/{name}.dcm"
    output:
        out = "{filepath}/{name}/{name}_method.json"
    run:
        shell(f"\"{DCM2NII_DIR/'enhDic2Nii.sh'}\" {input.dicom} || true")
        matlab_prefix = f"cd('{MATLAB_DIR.as_posix()}'); addMatlabPath;"
        shell(
            f"""matlab -batch "{matlab_prefix} niftiOrientation('{wildcards.filepath}/{wildcards.name}/{wildcards.name}');" """
        )

# MT MRI - generate MTR and MTsat maps
rule mtsat:
    input:
        "Data/{filepath}/{name}/{name}.nii.gz"  # MTw NIFTI (MT pulse on)
    output:
        "Data/{filepath}/{name}/{name}_mtsat.nii.gz"
    run:
        MT = "Data/" + wildcards.filepath + "/" + wildcards.name + "/" + wildcards.name
        if "ex_vivo" in wildcards.filepath:
            PD = "Data/" + wildcards.filepath + "/MToff_PD_GRE_3D_100x400_36A/MToff_PD_GRE_3D_100x400_36A"
            T1 = "Data/" + wildcards.filepath + "/MToff_T1_GRE_3D_100x400_36A/MToff_T1_GRE_3D_36A"
        else:
            PD = "Data/" + wildcards.filepath + "/MToff_PD_GRE_3D_150x400_12A/MToff_PD_GRE_3D_150x400_12A"
            T1 = "Data/" + wildcards.filepath + "/MToff_T1_GRE_3D_150x400_12A/MToff_T1_GRE_3D_150x400_12A"
        matlab_prefix = f"cd('{MATLAB_DIR.as_posix()}'); addMatlabPath;"
        shell(
            f"""matlab -batch "{matlab_prefix} nii2mtsat('{MT}','{PD}','{T1}','Data/{wildcards.filepath}/rpAFI_mouse_1/rpAFI_mouse_1','{MT}_mask',1);" """
        )
