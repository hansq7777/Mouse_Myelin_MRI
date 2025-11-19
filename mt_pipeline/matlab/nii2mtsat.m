function [mtr, mtsat] = nii2mtsat(name_MTon, name_MToff, name_T1w, maskName, gaussianFilter, name_B1)
%Function to convert NIFTIs to MTR and MTsat maps (input filenames must not
%include extensions such as .nii.gz,etc.)
%name_MToff: reference PDw scan with MT pulse off
%gaussianFilter = 0 or 1

if nargin < 6
    name_B1 = [];
end
if nargin < 5
    gaussianFilter = 0;
end
if nargin < 4
    maskName = [];
end
if nargin < 3
    name_T1w = [];
end

name_MTon = char(name_MTon);
name_MToff = char(name_MToff);
name_T1w = char(name_T1w);
name_B1 = char(name_B1);
maskName = char(maskName);

% Load data
if exist(sprintf('%s.nii.gz', name_MTon))
    ext = '.nii.gz';
elseif exist(sprintf('%s.nii', name_MTon))
    ext = '.nii';
else
    error('File extension error')
end


MTon_info = niftiinfo(sprintf('%s%s', name_MTon,ext));
MTon = single(niftiread(sprintf('%s%s', name_MTon,ext))).*MTon_info.MultiplicativeScaling;
refPD_info = niftiinfo(sprintf('%s%s', name_MToff,ext));
refPD = single(niftiread(sprintf('%s%s', name_MToff,ext))).*refPD_info.MultiplicativeScaling;

if ~isempty(name_T1w)
    refT1_info = niftiinfo(sprintf('%s%s', name_T1w,ext));
    refT1 = single(niftiread(sprintf('%s%s', name_T1w,ext))).*refT1_info.MultiplicativeScaling;

end
RFlocal_pre = [];
if ~isempty(name_B1)
    % Check if B1 is already an RFlocal map (3D). If so, load directly.
    b1_path = '';
    if exist(sprintf('%s.nii.gz', name_B1))
        b1_path = sprintf('%s.nii.gz', name_B1);
    elseif exist(sprintf('%s.nii', name_B1))
        b1_path = sprintf('%s.nii', name_B1);
    end
    if ~isempty(b1_path)
        tmp_info = niftiinfo(b1_path);
        if numel(tmp_info.ImageSize) == 3
            % Precomputed RFlocal map
            RFlocal_pre = single(niftiread(b1_path));
        end
    end

    if isempty(RFlocal_pre)
        % Need to derive B1 and RFlocal from a multi-volume B1 map
        if isempty(b1_path)
            if exist(sprintf('%s.nii.gz', name_B1))
                b1_path = sprintf('%s.nii.gz', name_B1);
            elseif exist(sprintf('%s.nii', name_B1))
                b1_path = sprintf('%s.nii', name_B1);
            else
                error('B1 file not found')
            end
        end
        % detect if fslroi/flirt available; if not, try WSL; if not, fallback MATLAB
        [st, ~] = system('which fslroi');
        use_fsl = (st == 0);
        if ~use_fsl
            % try WSL FSL
            [st, ~] = system('wsl which fslroi');
            use_fsl_wsl = (st == 0);
        else
            use_fsl_wsl = false;
        end

        if use_fsl
            [st, sysOut] = system(sprintf('fslroi "%s" "%s_vol2.nii.gz" 1 1',b1_path,name_B1)); if st; error(sysOut); end
            [st, sysOut] = system(sprintf('flirt -in "%s_vol2.nii.gz" -ref "%s.nii.gz" -out "%s_vol2_RS.nii.gz" -applyxfm',name_B1,name_MTon,name_B1)); if st; error(sysOut); end
            B1_info = niftiinfo(sprintf('%s_vol2_RS.nii.gz', name_B1));
            B1 = single(niftiread(sprintf('%s_vol2_RS.nii.gz', name_B1))).*B1_info.MultiplicativeScaling;
        elseif use_fsl_wsl
            % convert paths to WSL format
            b1_wsl = sprintf('/mnt/%c%s', lower(b1_path(1)), strrep(b1_path(3:end), '\','/'));
            mton_wsl = sprintf('/mnt/%c%s', lower(name_MTon(1)), strrep(name_MTon(3:end), '\','/'));
            out_base = sprintf('/mnt/%c%s', lower(name_B1(1)), strrep(name_B1(3:end), '\','/'));
            [st, sysOut] = system(sprintf('wsl fslroi \"%s\" \"%s_vol2.nii.gz\" 1 1',b1_wsl,out_base)); if st; error(sysOut); end
            [st, sysOut] = system(sprintf('wsl flirt -in \"%s_vol2.nii.gz\" -ref \"%s.nii.gz\" -out \"%s_vol2_RS.nii.gz\" -applyxfm',out_base,mton_wsl,out_base)); if st; error(sysOut); end
            B1_info = niftiinfo(sprintf('%s_vol2_RS.nii.gz', name_B1));
            B1 = single(niftiread(sprintf('%s_vol2_RS.nii.gz', name_B1))).*B1_info.MultiplicativeScaling;
        else
            % MATLAB-only fallback: load, take middle volume, resize to MTon dims if needed
            B1_info = niftiinfo(b1_path);
            B1_full = single(niftiread(b1_path));
            if ndims(B1_full) == 4 && size(B1_full,4) >= 2
                B1 = squeeze(B1_full(:,:,:,2));
            else
                error('B1 map does not contain volume 2');
            end
            if ~isequal(size(B1), size(MTon))
                B1 = imresize3(B1, size(MTon), 'linear');
            end
        end
    end
end

%apply Gaussian filter
if gaussianFilter
    for i = 1:size(MTon, 3)
        MTon(:,:,i) = imgaussfilt(MTon(:,:,i), 0.5, 'FilterSize', 3);
        refPD(:,:,i) = imgaussfilt(refPD(:,:,i), 0.5, 'FilterSize', 3);
	if ~isempty(name_T1w)
	    refT1(:,:,i) = imgaussfilt(refT1(:,:,i), 0.5, 'FilterSize', 3);
	end
	if ~isempty(name_B1) && isempty(RFlocal_pre)
	    B1(:,:,i) = imgaussfilt(B1(:,:,i), 0.5, 'FilterSize', 3);
	end
    end
end

% Calculate magnetization transfer ratio
mtr = (refPD-MTon)./refPD;

%mtc - may be used in registration
mtc = refPD - MTon;

% Calculate magnetization transfer saturation - call calcMTsat.m
dofilt = 0;
if ~isempty(name_T1w) 
    % Get flip angles and TR from json files
    MTon_json = fileread([name_MTon, '.json']);
    MTon_json = jsondecode(MTon_json);
    T1_json = fileread([name_T1w, '.json']);
    T1_json = jsondecode(T1_json);

    RFlocal = [];                      %RFlocal - relative local flip angle compared to nominal flip angle

    if ~isempty(name_B1)
        if ~isempty(RFlocal_pre)
            % Precomputed RFlocal map from NIfTI
            RFlocal = RFlocal_pre;
            % Optional per-slice CSV gating
            csvPath = sprintf('%s.csv', name_B1);
            if exist(csvPath, 'file')
                B1corr_slices = readmatrix(csvPath);
            else
                B1corr_slices = ones(size(RFlocal, 3), 1);
            end
            for z = 1:size(RFlocal, 3)
                if B1corr_slices(z) == 0  %don't apply B1 correction
                    RFlocal(:,:,z) = 1;
                end
            end
        else
            % Derive RFlocal from raw B1 map
            B1_json = fileread([name_B1, '.json']);
            B1_json = jsondecode(B1_json);
            RFlocal = B1/B1_json.FlipAngle;
            
            %check which slices B1 correction can be applied to based on
            %presence of banding artifact
            %If B1 correction CANNOT be applied, set RFlocal = 1
            csvPath = sprintf('%s.csv',name_B1);
            if exist(csvPath,'file')
                B1corr_slices = readmatrix(csvPath);
            else
                B1corr_slices = ones(size(RFlocal,3),1);
            end
            %go through each slice
            for z = 1:size(RFlocal, 3)
                if B1corr_slices(z) == 0  %don't apply B1 correction
                    RFlocal(:,:,z) = 1;
                end
            end
        end

        % Do the calculation with RFlocal
        mtsat = calcMTsat(refPD,refT1,MTon,pi/180*MTon_json.FlipAngle,pi/180*T1_json.FlipAngle,MTon_json.RepetitionTime,T1_json.RepetitionTime,RFlocal,dofilt);
    else
        % Do the calculation without RFlocal
        mtsat = calcMTsat(refPD,refT1,MTon,pi/180*MTon_json.FlipAngle,pi/180*T1_json.FlipAngle,MTon_json.RepetitionTime,T1_json.RepetitionTime,[],dofilt);
    end
end

% Make reasonable image bounds
mtr(mtr<0) = 0;
mtr(mtr>1) = 1;
if ~isempty(name_T1w) 
    mtsat(mtsat<0) = 0;
    mtsat(mtsat>0.1) = 0.1;
end

% Apply Mask 
if ~isempty(maskName)
    imMask = niftiread(sprintf('%s%s', maskName,ext));
    imMask = single(imMask);
    imMask(imMask > 0) = 1;
    mtr = mtr.*imMask;
    mtc = mtc.*imMask;
    if ~isempty(name_T1w)
        mtsat = mtsat.*imMask;
    end
end

% Save niftis
im_info = niftiinfo(sprintf('%s%s', name_MTon,ext));
im_info.MultiplicativeScaling = 1;
im_info.Datatype = 'single';
im_info.BitsPerPixel = 32;
im_info.raw.datatype = 16;
im_info.raw.bitpix = 32;
%
name_mtr = sprintf('%s_mtr',name_MTon);
name_mtc = sprintf('%s_mtc',name_MTon);
niftiwrite(single(mtr), name_mtr, im_info);
cmd = sprintf('gzip -f "%s.nii"', name_mtr); [st, sysOut] = system(cmd); if st; error(sysOut); end
niftiwrite(single(mtc), name_mtc, im_info);
cmd = sprintf('gzip -f "%s.nii"', name_mtc); [st, sysOut] = system(cmd); if st; error(sysOut); end
if ~isempty(name_T1w) 
    name_mtsat = sprintf('%s_mtsat',name_MTon);
    niftiwrite(single(mtsat), name_mtsat, im_info);
    cmd = sprintf('gzip -f "%s.nii"', name_mtsat); [st, sysOut] = system(cmd); if st; error(sysOut); end
end

end
