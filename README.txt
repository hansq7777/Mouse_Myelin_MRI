All code can be run by using the "rules" in the Snakefile. Scripts can also be run individually.

******************************************************************************************************************************************************************************************

Folders with code:

-dcm2nii: code to convert enhanced DICOMs to NIFTIs
-niiCombDiffAve: code to combine averages of diffusion MRI datasets (if they were collected separately on the scanner)
		This includes frequency drift correction and denoising, and for complex data partial Fourier recon and (optionally) refinement of the Nyquist N/2 ghost correction
-dMRIpreproc: code to preprocess diffusion MRI datasets
		Preprocessing steps include (in order):
		1. Denoising with mrtrix3 (this is turned off by default since complex denoising is done in niiCombDiffAve.mat, but it can be tunred on if needed)
		2. Gibbs ringing correction (with mrtrix3)
		3. TOPUP (from FSL) is applied to acquire reverse-blip susceptibility correction
		4. EDDY (from FSL) is run to apply the results from TOPUP and 
-scalarMapGen: code to output diffusion MRI and MT scalar maps

******************************************************************************************************************************************************************************************

Example Snakemake Usage to process dicoms all the way to scalar maps

The Snakemake rules used are listed directly below each command. Any number of files can be converted at once. In the example code below, the names 'filepath', 'mouse#', 'sex', 'timepoint', and 'dMRI_filename' are only placeholders and the user should input the actual filepath. Importantly, most of the code assumes that the dicom or NIFTI filename matches the name of the folder that it is in. All brain masks (for each MRI contrast) have been provided in the repository and the user should copy the masks to their respective folders, as the code assumes that these masks exist. Alternatively, the user can edit the code to run without masks.

Anatomical Data
To convert the anatomical dicoms (which include T2 and all MT related dicoms) to NIFTI format, the following Snakemake command can be used:
$ snakemake --cores 1 filepath/{mouse#1_sex/timepoint,mouse#2_sex/timepoint,mouse#3_sex/timepoint}/dicom_foldername/dicom_filename.json
[Rules: dcmTOnii_anat]
For example, a real use case of the above command, with the actual filepaths and filenames to acquire T2-weighted NIFTIs may be:
$ snakemake --cores 1 Data/{NR1_F/Day0,NR1_F/Day3,NR2_F/Day0}/T2_TurboRARE_AX150150500_A16/T2_TurboRARE_AX150150500_A16.json
Before acquiring MT metric maps, users must make a brain mask using the MT-weighted images (with software such as BrainSuite) and save the mask as “MTon_GRE_3D_150x400_12A_5uT_385FA_3500Hz_mask.nii.gz” in the folder “MTon_GRE_3D_150x400_12A_5uT_385FA_3500Hz.” As brain masks are also provided in the repository, users can also copy the mask, instead of creating a new one. To generate MT metric maps (MTR and MTsat), the following Snakemake command can be used:
$ snakemake --cores 1 filepath/{mouse#1_sex /timepoint,mouse#2_sex /timepoint,mouse#3_sex /timepoint}/ MTon_GRE_3D_150x400_12A_5uT_385FA_3500Hz/MTon_GRE_3D_150x400_12A_5uT_385FA_3500Hz_mtsat.nii.gz
[Rules: mtsat]

Diffusion MRI Data
To convert a number of dicoms to combined averages (in NIFTI format, with partial Fourier reconstruction, correction for frequency and signal drift, and denoising) and generate the initial dMRI brain mask (needed for preprocessing), the following Snakemake command can be used:
$ snakemake --cores 1 filepath/{mouse#1_sex /timepoint,mouse#2_sex /timepoint,mouse#3_sex /timepoint}/dMRI_filename/dMRI_filename{dicom_name1,dicom_name2,dicom_name3}_aveComb.nii.gz_aveComb_preproc_mask.nii.gz
[Rules: dcmTOnii_dMRI, combAve, get_preproc_mask]
The above command assumes that T2-weighted brain masks exist as “T2_TurboRARE_AX150150500_A16_mask.nii.gz” in the folder “T2_TurboRARE_AX150150500_A16,” as this mask is registered to dMRI space to create the initial dMRI brain mask. As the data acquired with reverse phase-encoding (“uFA_b0_reversePE” and  “OGSE_b0_reversePE”) do not require an initial mask, since they are combined with the larger datasets (“uFA_2Shapes_1A_3Rep_TR10” and “OGSE_5Shapes_1A_5Rep_TR10”), the command to convert dicoms with reverse phase-encoding to combined averages is:
$ snakemake --cores 1 filepath/{mouse#1_sex /timepoint,mouse#2_sex /timepoint,mouse#3_sex /timepoint}/dMRI_filename/dMRI_filename_aveComb.nii.gz
[Rules: dcmTOnii_dMRI, combAve]
After combined averages and initial dMRI brain masks are generated, preprocessing can be run by this command:
$ snakemake --cores 1 DiffusionDataPreproc/{mouse#1_sex /timepoint,mouse#2_sex /timepoint,mouse#3_sex /timepoint}/dMRI_filename/dMRI_filename_aveComb_preproc.nii.gz
[Rules: dMRIpreproc]
Note that the code assumes that the original NIFTI files are located in the “Data” folder and that FSL is being run from a singularity container. The user can change the code in “dMRIpreproc.sh” located in the folder “code_scidata_paper/dMRIpreproc” to align with their FSL environment. The above command will work with or without reverse phase-encoded data.
After the dMRI preprocessing step, final dMRI brain masks can be created, or the user can use the masks provided in the repository (“dMRI_filename_aveComb_preproc_mask_after.nii.gz”). The code assumes that the masks are named as they are in the repository. Alternatively, the user can acquire dMRI scalar maps without using brain masks by editing the code in the Snakefile. To acquire dMRI scalar maps, the following command can be run:
$ snakemake --cores 1 filepath/{mouse#1_sex /timepoint,mouse#2_sex /timepoint,mouse#3_sex /timepoint}/dMRI_filename/dMRI_filename_aveComb_preproc_mean_b0_Wmask.nii.gz
[Rules: get_dwimetric_maps]
The above command will generate scalar maps as well as a non-diffusion weighted (b0) NIFTI, averaged over all non-diffusion weighted volumes. This mean b0 NIFTI may be used to facilitate registration.

******************************************************************************************************************************************************************************************
