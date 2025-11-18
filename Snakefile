# Snakemake Pipeline for mouse data from 9.4 T - converts DICOMs to NIFTIs and generates scalar maps - OGSE dMRI, uFA dMRI, MT MRI
# (c) Naila Rahman, 2020-22


import numpy as np
import pathlib


######################################################################################################################################################################
#Anatomical - T2 and MT

#convert anatomical DICOMs to NIFTIs
rule dcmTOnii_anat:
    input:
        dicom = "{filepath}/{name}/{name}.dcm"
    output:
        out = "{filepath}/{name}/{name}_method.json"
    run:
        #convert to NIFTI
        shell("./dcm2nii/enhDic2Nii.sh {input.dicom} || true")
        #NIFTIs seem to be upside down, so re-orient NIFTIs to have cortex in the superior direction
        shell("""matlab -batch 'addMatlabPath(); niftiOrientation("{wildcards.filepath}/{wildcards.name}/{wildcards.name}");'""")

#MT MRI - generate MTR and MTsat maps
rule mtsat:
    input:
        "Data/{filepath}/{name}/{name}.nii.gz"                        #input is MTw NIFTI (with MT pulse on)
    output:
        "Data/{filepath}/{name}/{name}_mtsat.nii.gz"
    run:
        MT = "Data/" + wildcards.filepath + "/" + wildcards.name + "/" + wildcards.name
    	if "ex_vivo" in wildcards.filepath:
    	    PD = "Data/" + wildcards.filepath + "/MToff_PD_GRE_3D_100x400_36A/MToff_PD_GRE_3D_100x400_36A"
    	    T1 = "Data/" + wildcards.filepath + "/MToff_T1_GRE_3D_100x400_36A/MToff_T1_GRE_3D_100x400_36A"
        else:
            PD = "Data/" + wildcards.filepath + "/MToff_PD_GRE_3D_150x400_12A/MToff_PD_GRE_3D_150x400_12A"
    	    T1 = "Data/" + wildcards.filepath + "/MToff_T1_GRE_3D_150x400_12A/MToff_T1_GRE_3D_150x400_12A"
        shell("""matlab -batch 'addMatlabPath(); nii2mtsat("{MT}","{PD}","{T1}","Data/{wildcards.filepath}/rpAFI_mouse_1/rpAFI_mouse_1","{MT}_mask",1);'""")

######################################################################################################################################################################
#dMRI - OGSE and uFA

#convert dMRI DICOMs to NIFTIs
rule dcmTOnii_dMRI:
    input:
        dicom = "{filepath}/{dwiname}/{dwiname}.dcm"
    output:
        out = "{filepath}/{dwiname}/{dwiname}.bval"
    run:
        shell("./dcm2nii/enhDic2Nii.sh {input.dicom} || true")

#combine averages (if they were collected separately on the scanner)
rule combAve:
    input:
        "{filepath}/{dwiname}/{dwiname}.bval"
    output:
       "{filepath}/{dwiname}/{dwiname}_aveComb.nii.gz"
    run:
        #combine averages
        shell("""matlab -batch 'addMatlabPath(); opt.bthresh = []; opt.useGPU = 0; opt.dodenoise = 1; opt.gibbsAlpha = 1; niiCombDiffAve("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}",opt,1);' || true""")
        #reorient NIFTIs so cortex is in superior direction           
        shell("""matlab -batch 'addMatlabPath(); niftiOrientation("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb");' || true""")
        if "OGSE" in wildcards.dwiname:
            if ("b0" in wildcards.dwiname) == 0:
                #apply b-value calibration correction for ogse frequencies (calibration values based on a water phantom)
                bfactName = "bfact_water.txt"
                shell("""matlab -batch 'addMatlabPath(); bfact_corr("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb","{bfactName}");'""")

#only name_aveComb runs through this rule, not name_b0_aveComb (which are b0s acquired with reverse PE)
#get_preproc_mask brings T2 mask to b0 space (after dMRI averages have been combined, but before preprocessing)
#this initial mask is needed for EDDY in dMRIpreproc
rule get_preproc_mask:
    input:
        "{filepath}/{dwiname}/{dwiname}_aveComb.nii.gz"
    output:
        "{filepath}/{dwiname}/{dwiname}_aveComb_preproc_mask.nii.gz"
    run:
    	# get mean b0
    	shell("""matlab -batch 'addMatlabPath(); get_meanb0("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb");'""")
    	#register mean b0 to T2
    	b0 = wildcards.filepath + "/" + wildcards.dwiname + "/" + wildcards.dwiname + "_aveComb_mean_b0"
    	if "ex_vivo" in wildcards.filepath:
    	    T2 = wildcards.filepath + "/T2_TurboRARE_AX100100500_A48/T2_TurboRARE_AX100100500_A48"
        else:
    	    T2 = wildcards.filepath + "/T2_TurboRARE_AX150150500_A16/T2_TurboRARE_AX150150500_A16"
    	shell("antsRegistration -d 3 -r [{b0}.nii.gz,{T2}.nii.gz,1] -m MI[{b0}.nii.gz,{T2}.nii.gz,1,32] -t Affine[0.1] -c 10000x10000x10000x10000x10000 -s 0.8x0.6x0.4x0.2x0mm -f 5x4x3x2x1 -l 1 -o [{b0}_transformT2affine,{b0}_warpedT2affine.nii.gz,{b0}_inverseWarpedT2affine.nii.gz]")
    	#apply transforms
    	shell("antsApplyTransforms -d 3 -i {T2}_mask.nii.gz -r {b0}.nii.gz -o {b0}_mask_warped_binary.nii.gz -n NearestNeighbor -t [{b0}_transformT2affine0GenericAffine.mat,1]")
    	#rename mask
    	shell("cp {b0}_mask_warped_binary.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask.nii.gz")
        
##################################################################################################################
        
#preprocess niftis - includes denoising, gibbs ringing correction, and susceptibility and eddy current induced distortion correction
rule dMRIpreproc:
    input:
        "Data/{filepath}/{dwiname}/{dwiname}_aveComb_preproc_mask.nii.gz"
    output:
        "DiffusionDataPreproc/{filepath}/{dwiname}/{dwiname}_aveComb_preproc.nii.gz"
    run:
        #mkdir in DataPreproc
        shell("mkdir -p DiffusionDataPreproc/{wildcards.filepath}/{wildcards.dwiname}")
        #cp mask to correct dir
        shell("cp Data/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask.nii.gz DiffusionDataPreproc/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask.nii.gz")
        #run dMRIpreproc.sh
        #if b0 file (with reverse phase encoding) exists, then topup will be run
        b0_name = "Data/" + wildcards.filepath + "/" + wildcards.dwiname + "_b0/" + wildcards.dwiname + "_b0_aveComb.nii.gz"
        b0_file = pathlib.Path(b0_name)
        if b0_file.exists():					#run topup
            shell("./dMRIpreproc/dMRIpreproc.sh DiffusionDataPreproc/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc Data/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb Data/{wildcards.filepath}/{wildcards.dwiname}_b0_reversePE/{wildcards.dwiname}_b0_reversePE_aveComb 0")
        else:                                                  #topup will not be run
            shell("./dMRIpreproc/dMRIpreproc.sh DiffusionDataPreproc/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc Data/{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb")

        
#generate scalar maps for dMRI data
rule get_dwimetric_maps:
    input:
        "{filepath}/{dwiname}/{dwiname}_aveComb_preproc.bvec"
    output:
        "{filepath}/{dwiname}/{dwiname}_aveComb_preproc_delFA.nii.gz"
    run:
        if "uFA" in wildcards.dwiname:
            # isIso: list specifying STE (spherical tensor encoding) volumes
            if "ex_vivo" in wildcards.filepath:
                isIso = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            else:
                isIso = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
            # call nii2uFA.m
            shell("""matlab -batch 'addMatlabPath(); nii2uFA("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc",[{isIso}],"{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after",200);'""")

        else: #OGSE
            if "ex_vivo" in wildcards.filepath:
                #OGSE frequencies
                freqStrings = ["000","050","080","115","150"]
                freqList = [0, 50, 80, 115, 150]
                #copy mask from uFA folder - no need to make new mask for OGSE
                shell("cp {wildcards.filepath}/uFA_res130150500/uFA_res130150500_aveComb_preproc_mask_after.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after.nii.gz")
            else:
                #copy mask from uFA folder - no need to make new mask for OGSE
                shell("cp {wildcards.filepath}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_aveComb_preproc_mask_after.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after.nii.gz")
                # OGSE frequencies
                freqStrings = ["000","050","100","145","190"]
                freqList = [0, 50, 100, 145, 190]
            
            # ogse2freq.m separates OGSE frequencies and saves them as separate NIFTIs 
            shell("""matlab -batch 'addMatlabPath(); ogse2freq("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc",[{freqList}],10);'""")
            # for each OGSE frequency, call nii2uFA.m, which also outputs DTI metrics
            for i in range(0, len(freqStrings)):
                freq = freqStrings[i];
                shell("""matlab -batch 'addMatlabPath(); nii2uFA("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freq}","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freq}",[],"{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after",200);'""")

            # get MD difference and diffusion dispersion maps (least square fit of MD to frequency) - MD2Gmap.m
            shell("fslmaths {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[4]}_MD.nii.gz -sub {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[0]}_MD.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_delMD.nii.gz")
            shell("fslmaths {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[4]}_FA.nii.gz -sub {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[0]}_FA.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_delFA.nii.gz")
            shell("""matlab -batch 'addMatlabPath(); MDtoGmap("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[0]}_MD","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[1]}_MD","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[2]}_MD","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[3]}_MD","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_f{freqStrings[4]}_MD","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after",[{freqList}]);'""")

        #get mean b0
        shell("""matlab -batch 'addMatlabPath(); get_meanb0("{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc","{wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc");'""")

        #apply mask to mean b0
        shell("fslmaths {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mean_b0.nii.gz -mas {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mask_after.nii.gz {wildcards.filepath}/{wildcards.dwiname}/{wildcards.dwiname}_aveComb_preproc_mean_b0_Wmask.nii.gz")

######################################################################################################################################################################
#Registration Rules
#For in vivo data only
#These outputs are not included in the repository.

#Registration - dwi metric maps to FA template to T2 template to atlas
rule dwiTOatlas:
    input:
        "{filepath}/{mouse}_{gender}/{time}/{contrast}/{contrast}_aveComb_preproc_{metric}.nii.gz"
    output:
        "{filepath}/{mouse}_{gender}/{time}/{contrast}/{contrast}_aveComb_preproc_{metric}_warpedTOatlas.nii.gz"
    run:
        DWImetric = wildcards.filepath + "/" + wildcards.mouse + "_" + wildcards.gender + "/" + wildcards.time + "/" + wildcards.contrast + "/" + wildcards.contrast + "_aveComb_preproc_" + wildcards.metric
        atlas = "Registration/atlas/TMBTA_Brain_Template_reorient_smoothed0_2_RS_Gaussian"
        T2_atlas_transform = "Registration/T2template_to_atlas/T2_atlas_SynMI0_00005_transform"
        FA_T2_template_transform = "Registration/FAtemplate_to_T2template/FA_T2_SynMI0_005_transform"
        DWI_FA_template_transform = "Registration/ANTStemplate_FA/template_b2000_FA_"
        shell("fslchpixdim {DWImetric}.nii.gz 2 2 5")
        shell("antsApplyTransforms -d 3 -i {DWImetric}.nii.gz -r {atlas}.nii.gz -o {DWImetric}_warpedTOatlas.nii.gz -t {T2_atlas_transform}1Warp.nii.gz -t {T2_atlas_transform}0GenericAffine.mat -t {FA_T2_template_transform}1Warp.nii.gz -t {FA_T2_template_transform}0GenericAffine.mat -t {DWI_FA_template_transform}{wildcards.mouse}_{wildcards.gender}_{wildcards.time}1Warp.nii.gz -t {DWI_FA_template_transform}{wildcards.mouse}_{wildcards.gender}_{wildcards.time}0GenericAffine.mat -f 0 -v 1")
        
        
#Registration - mtr and mtsat maps to MT_on template to T2 template to atlas
rule mtrTOatlas:
    input:
        "{filepath}/{mouse}_{gender}/{time}/{contrast}/{contrast}_{metric}.nii.gz"
    output:
        "{filepath}/{mouse}_{gender}/{time}/{contrast}/{contrast}_{metric}_warpedTOatlas.nii.gz"
    run:
        MTmetric = wildcards.filepath + "/" + wildcards.mouse + "_" + wildcards.gender + "/" + wildcards.time + "/" + wildcards.contrast + "/" + wildcards.contrast + "_" + wildcards.metric
        atlas = "Registration/atlas/TMBTA_Brain_Template_reorient_smoothed0_2_RS_Gaussian"
        T2_atlas_transform = "Registration/T2template_to_atlas/T2_atlas_SynMI0_00005_transform"
        MT_T2_template_transform = "Registration/MTtemplate_to_T2template/MT_T2_SynCI0.005_transform"
        mtr_MT_template_transform = "Registration/ANTStemplate_MT/template_MTon_"
        shell("fslchpixdim {MTmetric}.nii.gz 1.5 1.5 4")
        shell("antsApplyTransforms -d 3 -i {MTmetric}.nii.gz -r {atlas}.nii.gz -o {MTmetric}_warpedTOatlas.nii.gz -t {T2_atlas_transform}1Warp.nii.gz -t {T2_atlas_transform}0GenericAffine.mat -t {MT_T2_template_transform}1Warp.nii.gz -t {MT_T2_template_transform}0GenericAffine.mat -t {mtr_MT_template_transform}{wildcards.mouse}_{wildcards.gender}_{wildcards.time}1Warp.nii.gz -t {mtr_MT_template_transform}{wildcards.mouse}_{wildcards.gender}_{wildcards.time}0GenericAffine.mat -f 0 -v 1")



# Register b0 to b0 - can be used for test-retest analysis - can apply registration transforms to scalar maps
rule reg_b0_b0:
    input:
        "DiffusionDataPreproc/{filepath}/{timepoint}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_aveComb_preproc_MD.nii.gz"
    output:
        "DiffusionDataPreproc/{filepath}/{timepoint}_reg_chosenb0/{timepoint}_chosenb0_Warped.nii.gz"
    run:
        b0_chosen = "DiffusionDataPreproc/NR6_M/Day2/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_aveComb_preproc_mean_b0_Wmask"
        b0 = "DiffusionDataPreproc/" + wildcards.filepath + "/" + wildcards.timepoint + "/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_aveComb_preproc_mean_b0_Wmask"
        output = "DiffusionDataPreproc/" + wildcards.filepath +  "/" + wildcards.timepoint + "_reg_chosenb0/" + wildcards.timepoint + "_chosenb0"
        #register
        shell("antsRegistration -d 3 -r [{b0_chosen}.nii.gz,{b0}.nii.gz,1] -m MI[{b0_chosen}.nii.gz,{b0}.nii.gz,1,32] -t Affine[0.1] -c 10000x10000x10000x10000x10000 -s 0.8x0.6x0.4x0.2x0mm -f 5x4x3x2x1 -l 1 -r [{b0_chosen}.nii.gz,{b0}.nii.gz,1] -m MI[{b0_chosen}.nii.gz,{b0}.nii.gz,1,32] -t SyN[0.005] -c 50x35x15 -f 3x2x1 -s 0.4x0.2x0mm -n BSpline -u true -l 1 -o [{output}_transform,{output}_Warped.nii.gz,{output}_InverseWarped.nii.gz] -v 1")

######################################################################################################################################################################
#Miscellaneous rules

#apply mask
rule applyMask:
    input:
        "{filepath}/{name}/{name}.nii.gz"
    output:
        "{filepath}/{name}/{name}_Wmask.nii.gz"
    run:
        shell("fslmaths {wildcards.filepath}/{wildcards.name}/{wildcards.name}.nii.gz -mas {wildcards.filepath}/{wildcards.name}/{wildcards.name}_mask.nii.gz {wildcards.filepath}/{wildcards.name}/{wildcards.name}_Wmask.nii.gz")

#get mean b0
rule getmeanb0:
    input:
        "{filepath}/{name}/{name}_aveComb.nii.gz"
    output:
        "{filepath}/{name}/{name}_aveComb_mean_b0.nii.gz"
    run:
        shell("""matlab -batch 'addMatlabPath(); get_meanb0("{wildcards.filepath}/{wildcards.name}/{wildcards.name}_aveComb","{wildcards.filepath}/{wildcards.name}/{wildcards.name}_aveComb");'""")
        
#copy .mat and warp files from template folder to correct folders - dMRI (these are warp files for individual FA maps to the template FA map)
#The user can change filenames to apply this to the MT data
rule send_warp_files_dwi:
    input:
        "Registration/ANTStemplate_FA/template_b2000_FA_{mouse}_{gender}_{time}*GenericAffine.mat"
    output:
        "DiffusionDataPreproc/{mouse}_{gender}/{time}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_b2000_FA_Warp.nii.gz"
    run:
        shell("mv Registration/ANTStemplate_FA/template_b2000_FA_{wildcards.mouse}_{wildcards.gender}_{wildcards.time}*GenericAffine.mat DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_b2000_FA_GenericAffine.mat")
        shell("mv Registration/ANTStemplate_FA/template_b2000_FA_{wildcards.mouse}_{wildcards.gender}_{wildcards.time}*Warp.nii.gz DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10")
        #renaming the files
        shell("mv DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10/template_b2000_FA_{wildcards.mouse}_{wildcards.gender}_{wildcards.time}*InverseWarp.nii.gz DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_b2000_FA_InverseWarp.nii.gz")
	shell("mv DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10/template_b2000_FA_{wildcards.mouse}_{wildcards.gender}_{wildcards.time}*Warp.nii.gz DiffusionDataPreproc/{wildcards.mouse}_{wildcards.gender}/{wildcards.time}/uFA_2Shapes_1A_3Rep_TR10/uFA_2Shapes_1A_3Rep_TR10_b2000_FA_Warp.nii.gz")
