function [uA, MD, uFA, Kiso, Klin, b0Est] = nii2uFA(niftiName, bvalName, isIso, maskName, bthresh, savedir, useOLS)
% Function to convert preprocessed NIFTI data to uFA. Filenames must not include extensions (.nii.gz is assumed for nifti, and .bval, .bvec, .json for support files)
%
% [uA, MD, uFA, Kiso, Klin] = nii2uFA(niftiName, bvalName, isIso, maskName, bthresh)
%
% maskName: set to 0 to use no mask; set to a value to mask with that
% thresh; set to [] to try to automatically set mask.
%
%isIso = list identifying STE volumes
%savedir = directory to save output files
%
% uA = micro-anisotropy  
% MD = mean diffusivity
% Kiso = isotropic kurtosis from powder average STE signal
% Klin = linear kurtosis from powder LTE average signal
% b0Est = what a T2 weighted image with b perfectly equal to 0 should look
%   like (this drops out of the fitting process for free)
%

niftiName = char(niftiName);
bvalName = char(bvalName);
maskName = char(maskName);

path_mrtrix = '/srv/software/mrtrix_3.0.2';

% Initialize outputs
MD = [];
Kiso = [];
Klin = [];
b0Est = [];
uFA = [];
uA = [];
muIso = [];

if nargin<2 || isempty(bvalName)
    bvalName = niftiName;
end
if nargin<3
    isIso = [];
end
if nargin<4
    maskName = [];
end
if nargin<5 || isempty(bthresh)
    bthresh = 55;
end
if nargin<6 
    savedir = [];
end
if nargin<7 || isempty(useOLS)
    % Use ordinary least squares instead of non-negative least squares
    useOLS = 0;
end

% Load data
if iscell(niftiName)
    % Allow multiple averages to be inputted. NB: all averages must have identical scan parameters!
    if exist([niftiName{1},'.nii.gz'],'file')
        ext = '.nii.gz';
    elseif exist([niftiName{1},'.nii'],'file')
        ext = '.nii';
    else
        error('File does not exist');
    end
    im = 0;
    for n=1:length(niftiName)
        im = im + single(niftiread(sprintf('%s%s', niftiName{n},ext)))/length(niftiName);
    end
    niftiName = niftiName{1};
    if iscell(bvalName)
        bvalName = bvalName{1};
    end
elseif isa(niftiName,'char')
    if exist([niftiName,'.nii.gz'],'file')
        ext = '.nii.gz';
    elseif exist([niftiName,'.nii'],'file')
        ext = '.nii';
    else
        error('File does not exist');
    end
    im = single(niftiread(sprintf('%s%s', niftiName,ext)));
else
    im = niftiName;
end

% Set save name
if ~isempty(savedir)
    [~,NAME,~] = fileparts(niftiName);
    savename = [savedir,filesep,NAME];
else
    savename = niftiName;
end

% Parse for b-matrix input
if isa(niftiName,'char') && exist([bvalName, '.bmat'], 'file')
    % File should have all matrix entries in a 9 by Ndir array 
    bmat = dlmread([bvalName, '.bmat']);
    Ndir = size(bmat,2);
    bmat = reshape(bmat', Ndir, 3, 3);
    bval = zeros(Ndir,1);
    isIso = false(Ndir,1);
    for n=1:Ndir
        bval(n) = trace(squeeze(bmat(n,:,:)));
        isIso(n) = round(rank(squeeze(bmat(n,:,:)), 0.25*bval(n))) == 3;
    end
else
    bval = dlmread([bvalName, '.bval']);
    bval = bval(:);
    isIso = isIso(:);
end
if isempty(isIso)
    isIso = false(size(bval));
end

% Find bval shells
bshells = bval(1);
for n=2:length(bval)
    if ~any(abs(bval(n)-bshells) < bthresh)
        bshells = cat(1,bshells,bval(n));
    end
end
for n=1:length(bshells)
    bshells(n) = mean(bval(abs(bval-bshells(n))<bthresh));
end
bshells = sort(bshells,'ascend');

fprintf('%d shells found\n',length(bshells));

% Determine which shells have iso or lin
isoShells = zeros(size(bshells));
linShells = zeros(size(bshells));
bisoShells = zeros(size(bshells));
blinShells = zeros(size(bshells));
isIsoShells = false(length(bval),length(bshells));
isLinShells = false(length(bval),length(bshells));
for n=1:length(bshells)
    if bshells(n) < bthresh
        isoShells(n) = 1;
        linShells(n) = 1;
        bisoShells(n) = bshells(n);
        blinShells(n) = bshells(n);
        isIsoShells(:,n) = bval < bthresh;
        isLinShells(:,n) = bval < bthresh;
    else
        if sum( and( isIso,abs(bval-bshells(n))<bthresh) ) > 0
            isoShells(n) = 1;
            %bisoShells(n) = bshells(n); 
            isIsoShells(:,n) = and(isIso, abs(bval-bshells(n))<bthresh);
            bisoShells(n) = mean(bval(isIsoShells(:,n)));
        end
        if sum( and(~isIso,abs(bval-bshells(n))<bthresh) ) > 0
            linShells(n) = 1;
            %blinShells(n) = bshells(n);
            isLinShells(:,n) = and(~isIso, abs(bval-bshells(n))<bthresh);
            blinShells(n) = mean(bval(isLinShells(:,n)));
        end
    end
end

% Prep data for calculations. Take powder averages, and sort in order of b-value
valMask = [];
sz = size(im);
if sum(isoShells)>0
    yiso = [];
    for n=1:length(bshells)
        if isoShells(n)
            yiso = cat(4, yiso, mean(im(:,:,:,isIsoShells(:,n)),4));
        else
            yiso = cat(4, yiso, zeros(sz(1:3)));
        end
    end
    valMask = sum(abs(yiso),4)>0;
    yiso = log(yiso);
end
corrB = 0;
if ~corrB
    if sum(linShells)>0
        ylin = [];
        for n=1:length(bshells)
            if linShells(n)
                ylin = cat(4, ylin, mean(im(:,:,:,isLinShells(:,n)),4));
            else
                ylin = cat(4, ylin, zeros(sz(1:3)));
            end
        end
        if isempty(valMask)
            valMask = sum(abs(ylin),4)>0;
        else
            valMask = or(valMask, sum(abs(ylin),4)>0);
        end
        ylin = log(ylin);
    end
else
    error('TODO: refactor and test')
    % Take varying b-values into account (from cross terms). Assume Gaussian
    % diffusion between this and next lowest shell
    if sum(linShells)>0
        ylin = mean(im(:,:,:,isLinShells(:,1)),4);
        if isempty(ylin)
            yref = yiso(:,:,:,1);
        else
            yref = ylin;
        end
        for ns=2:length(bshells)
            tmp = im(:,:,:,isLinShells(:,ns));
            if ~isempty(tmp)
                if corrB
                    % Take varying b-values into account (from cross terms). Assume Gaussian
                    % diffusion between this and next lowest shell
                    btmp = bval(isLinShells(:,ns));
                    p = btmp/blinShells(ns);
                    for n=1:size(tmp,4)
                        tmp(:,:,:,n) = ((tmp(:,:,:,n)./yref).^p(n)).*yref;
                    end
                end
                yref = mean(tmp,4);
                ylin = cat(4, ylin, yref);
            else
                yref = yiso(:,:,:,ns);
            end     
        end
        ylin = log(ylin);
    end
end

% Remove inf voxels
yiso(~repmat(valMask, [1 1 1 size(yiso,4)])) = 0;
ylin(~repmat(valMask, [1 1 1 size(ylin,4)])) = 0;

% Compute uA
if any(isIso)
    sharedShell = and(isoShells,linShells);
    sharedShell = find(sharedShell,1,'last'); % highest b-value
    uA2 = ylin(:,:,:,sharedShell)/blinShells(sharedShell)^2 - ...
        yiso(:,:,:,sharedShell)/bisoShells(sharedShell)^2;
    uA = abs(sqrt(uA2));
end

% Remove empty entries
yiso = yiso(:,:,:,isoShells>0);
ylin = ylin(:,:,:,linShells>0);
bisoShells = bisoShells(isoShells>0);
blinShells = blinShells(linShells>0);

if ~isempty(maskName) && ischar(maskName)
    imMask = niftiread(sprintf('%s.nii.gz', maskName));
    imMask = single(imMask);
    imMask = flip(imMask, 1);
    imMask (imMask > 0) = 1;
elseif maskName <= 0
    imMask = [];
else
    [~,mI] = min(bval);
    im_a = abs(im(:,:,:,mI));
    if ~isempty(maskName)
        thresh = maskName;
    else
        threshSNR = 2;
        [counts,edges] = histcounts(im_a(:),10000);
        counts(1) = 0; % Remove voxels with value = 0
        [~,counts] = max(counts);
        thresh = threshSNR*0.5*(edges(counts)+edges(counts+1)); % The peak will occur at the noise level. Scaling by threshSNR assumes only voxels with SNR>threshSNR are relevant
    end
    imMask = abs(im_a) > thresh;
    % Filter mask
    filtsz = round(size(im_a,1)/25);
    filtsz = filtsz + (1-mod(filtsz,2));
    filtM = -floor(filtsz/2):floor(filtsz/2);
    [X,Y] = meshgrid(filtM);
    filtM = sqrt(X.^2+Y.^2);
    filtM = filtM <= floor(filtsz/2)+0.2;
    filtM = repmat(filtM, [1 1 3]);
    imMask = convn(imMask,filtM,'same');
    imMask = imMask > sum(filtM(:))/2;
    % The end slices do not work well
    imMask(:,:,1) = imMask(:,:,2);
    imMask(:,:,end) = imMask(:,:,end-1);
    clear im_a
end

% Perform other parameter fits
MD = [];
if sum(isoShells)>2 && sum(linShells)==0
    % Estimate kurtosis from iso scan
    [MD, Kiso, b0Est] = computeKurtosis(bisoShells, yiso);
elseif sum(isoShells)==0 && sum(linShells)>2
    [MD, Klin, b0Est] = computeKurtosis(bisoShells, yiso);
elseif or(sum(isoShells)>2 && sum(linShells)>0, sum(linShells)>2 && sum(isoShells)>0)
    % Scale b-values to be in units of ms/um^2
    bisoShells = bisoShells/1000;
    blinShells = blinShells/1000;
    bthresh = bthresh/1000;
    % Jointly estimate kurtosis from iso and lin powder averages
    A = [ones(length(bisoShells),1), -bisoShells(:), bisoShells(:).^2, zeros(length(bisoShells),1), zeros(length(bisoShells),1)];
    blinShells_a = blinShells;
    if (blinShells_a(1) < bthresh) && (bisoShells(1) < bthresh)
        % Avoid duplication of b=0 acquisitions
        blinShells_a = blinShells_a(2:end);
        ylin = ylin(:,:,:,2:end); 
    end
    A = cat(1, A, [ones(length(blinShells_a),1), zeros(length(blinShells_a),1), zeros(length(blinShells_a),1), -blinShells_a(:), blinShells_a(:).^2]);
    lambda_MD = 1e6;
    lambda_uA = 0; %1e6;
    lambda_tik = 0; %1e-1; % I thought this would decrease voxels with super high K, but it doesn't. It seems they stem from spots with a low MD, which happens near CSF.
    if lambda_MD>0
        A = cat(1, A, lambda_MD*[0 1 0 -1 0]); % Set that MDlin = MDiso as a soft constraint. Lambda weights the contraint
    end
    if lambda_uA>0
        A = cat(1, A, lambda_uA*[0 -1 0 1 0]); % Set that difference of 2rd order cumulants = uA2, which is how above eq for uA2 was derived
    end
    if lambda_tik>0
        % tikhonov reg on kurtosis params. Could also regularize against a smoothed version...
        A = cat(1,A,lambda_tik*[0 0 0 0 1]);
    end
    AtA = A'*A;
    y = permute(yiso, [4 1 2 3]);
    y1 = permute(ylin, [4 1 2 3]);
    y = cat(1, y, y1); clear('y1');
    if lambda_MD>0
        y = cat(1, y, zeros([1 size(y,2) size(y,3) size(y,4)],'like',y));  % Append zeros for MD constraint
    end
    if lambda_uA>0
        y = cat(1, y, permute(uA2, [4 1 2 3])); % Append uA^2 for uA constraint
    end
    if lambda_tik>0
        y = cat(1, y, zeros([1 size(y,2) size(y,3) size(y,4)],'like',y));  % Append zeros for tikhonov regularization of Kurtosis
    end
    szy = size(y);
    x = zeros([size(A,2), szy(2:end)]);
    y = double(real(y));
    options = optimset('Display','off');
    tic1 = tic;
    if useOLS
        fprintf('Estimating Kurtosis using ordinary least squares...')
    else
        fprintf('Estimating Kurtosis using non-negative least squares...')
    end
    pA = pinv(A);
    for n=1:size(x(:,:),2)
        if isempty(imMask) || (imMask(n)>0.5)
            if useOLS
                x(:,n) = pA*y(:,n);
            else
                x(:,n) = lsqnonneg(A,y(:,n),options);
            end
        end
    end
    fprintf('took %d sec\n', round(toc(tic1)));
    szim = size(im);
    x = reshape(x, [5, szim(1:3)]);
    MDiso = squeeze(x(2,:,:,:));
    MDlin = squeeze(x(4,:,:,:));
    MD = 0.5*(MDiso+MDlin); 
    muIso = 2*squeeze(x(3,:,:,:));
    Kiso = 6*squeeze(x(3,:,:,:))./MD.^2;
    Klin = 6*squeeze(x(5,:,:,:))./MD.^2;
    b0Est = exp(squeeze(x(1,:,:,:)));
    uA2_b = squeeze(x(5,:,:,:)) - squeeze(x(3,:,:,:));
    % Filter out implausible huge K
    Kiso(Kiso(:)>3) = 3;
    Klin(Klin(:)>3) = 3;
    % Scale b-values to be in units of s/mm^2
    bisoShells = bisoShells*1000;
    blinShells = blinShells*1000;
    bthresh = bthresh*1000;
    MD = MD/1000;
    uA2_b = uA2_b/1e6;
elseif sum(isoShells) > 1
    % Estimate MD from iso scan
    MD = (yiso(:,:,:,1) - yiso(:,:,:,2))/(bisoShells(2)-bisoShells(1));
elseif sum(linShells) > 1
    MD = (ylin(:,:,:,1) - ylin(:,:,:,2))/(blinShells(2)-blinShells(1));
end
clear y

if exist('lambda_uA','var') && lambda_uA == 0
    uA2 = uA2_b;
    uA = abs(sqrt(uA2));
    clear uA2_b
end

% Find uFA if possible from data
uFA_filt = 1.5;
if ~isempty(MD) && ~isempty(uA)
    uFA = 1.5*uA2./(uA2+0.2*MD.^2);
    % Filter out implausible values
    uFA(uFA(:)<0) = 0;
    uFA(uFA(:)>uFA_filt) = uFA_filt;
    % Take to square root
    uFA = sqrt(uFA);
end

% Compute DTI metrics
if sum(linShells)>1 && isa(niftiName, 'char')
    shellInds = find(linShells,2,'first'); % Only use the lowest shells
    coordInds = find(and(or(~isIso, bval<bthresh), bval<max(blinShells(shellInds))+bthresh));
    coordInds = sprintf('%d,', coordInds-1); 
    coordInds = ['3 ', coordInds(1:end-1)]; % Remove newline
    cmd = sprintf('%s/mrconvert %s%s -force -coord %s -fslgrad %s.bvec %s.bval tmp.mif', path_mrtrix,niftiName, ext, coordInds, bvalName, bvalName);
    [st, sysOut] = system(cmd); if st; error(sysOut); end
    if ~isempty(maskName)
        cmd = sprintf('%s/dwi2tensor tmp.mif -force -mask %s.nii.gz',path_mrtrix,maskName);
    else
        cmd = sprintf('%s/dwi2tensor tmp.mif -force',path_mrtrix);
    end
    if 0%sum(linShells)>2
        % This worsens the DTI quality in some cases, so not always best to
        % do
        cmd = sprintf('%s -ols -iter 4 -dkt %s_dkt.nii.gz -config BZeroThreshold %d tmp2.mif', cmd, savename, round(bthresh));
    else
        cmd = sprintf('%s -ols -iter 4 -config BZeroThreshold %d tmp2.mif', cmd, round(bthresh));
    end
    [st, sysOut] = system(cmd); if st; error(sysOut); end
    cmd = sprintf('%s/tensor2metric tmp2.mif -force -fa %s_FA.nii.gz -vector %s_FAvec.nii.gz -ad %s_AD.nii.gz -rd %s_RD.nii.gz -adc %s_ADC.nii.gz', path_mrtrix, savename, savename,savename, savename, savename);
    if ~isempty(maskName)
        cmd = [cmd, sprintf(' -mask %s.nii.gz', maskName)];
    end
    [st, sysOut] = system(cmd); if st; error(sysOut); end
    delete tmp.mif tmp2.mif 
    
    if ~isempty(uA)             %also compute DTI metrics for b2000
        shellInds = [1; 3]; % b2000
        coordInds = find(and(or(~isIso, bval<bthresh), bval<max(blinShells(shellInds))+bthresh));
        coordInds = sprintf('%d,', coordInds-1); 
        coordInds = ['3 ', coordInds(1:end-1)];
        cmd = sprintf('%s/mrconvert %s.nii.gz -force -coord %s -fslgrad %s.bvec %s.bval tmp.mif', path_mrtrix, niftiName, coordInds, bvalName, bvalName);
        [st, sysOut] = system(cmd); if st; error(sysOut); end
        cmd = sprintf('%s/dwi2tensor -ols tmp.mif -force -iter 4 -config BZeroThreshold %d tmp2.mif', path_mrtrix, round(bthresh));
        [st, sysOut] = system(cmd); if st; error(sysOut); end
        cmd = sprintf('%s/tensor2metric tmp2.mif -force -fa %s_b2000_FA.nii.gz -vector %s_b2000_FAvec.nii.gz', path_mrtrix, niftiName, niftiName);
        if ~isempty(maskName)
            cmd = [cmd, sprintf(' -mask %s.nii.gz', maskName)];
        end
        [st, sysOut] = system(cmd); if st; error(sysOut); end
        delete tmp.mif tmp2.mif
    end
end


% Mask 
if ~isempty(imMask)
    if ~isempty(uA)
        uA = uA.*imMask;
    end
    if ~isempty(MD)
        MD = MD.*imMask;
    end
    if ~isempty(uFA)
        uFA = uFA.*imMask;
    end
    if ~isempty(Kiso)
         Kiso = Kiso.*imMask;
    end
    if ~isempty(Klin)
         Klin = Klin.*imMask;
    end
    if ~isempty(muIso)
         muIso = muIso.*imMask;
    end
end

% Save result
if isa(savename, 'char')
    im_info = niftiinfo(sprintf('%s%s', niftiName, ext));
    im_info.PixelDimensions = im_info.PixelDimensions(1:3);
    im_info.ImageSize = im_info.ImageSize(1:3);
    im_info.raw.dim(1) = 3;
    im_info.raw.dim(5) = 1;
    im_info.raw.pixdim(5) = 0;
    im_info.raw.dim_info = ' ';
    %
    im_info.Datatype = 'single';
    im_info.BitsPerPixel = 32;
    im_info.raw.datatype = 16;
    im_info.raw.bitpix = 32;
    % niftiwrite doesn't support names with periods
    savename(savename == '.') = '_';
    %
    %DWI
    DWI_lin = exp(ylin(:,:,:,2));
    niftiwrite(single(DWI_lin), sprintf('%s_DWI_lin', niftiName), im_info, 'Compressed', true);
    %
    if ~isempty(uA)
        niftiwrite(single(uA), sprintf('%s_uA', niftiName), im_info, 'Compressed', true);
    end
    if ~isempty(MD)
        niftiwrite(single(MD), sprintf('%s_MD', niftiName), im_info, 'Compressed', true);
    end
    if ~isempty(uFA)
        niftiwrite(single(uFA), sprintf('%s_uFA', niftiName), im_info, 'Compressed', true);
    end
    if ~isempty(Kiso)
        niftiwrite(single(Kiso), sprintf('%s_Kiso', niftiName), im_info, 'Compressed', true);
    end
    if ~isempty(muIso)
        niftiwrite(single(muIso), sprintf('%s_muIso', niftiName), im_info, 'Compressed', true);
    end
    if ~isempty(Klin)
        niftiwrite(single(Klin), sprintf('%s_Klin', niftiName), im_info,'Compressed', true );
    end
    if ~isempty(b0Est)
        niftiwrite(single(b0Est), sprintf('%s_b0Est', niftiName), im_info, 'Compressed', true);
    end
    dlmwrite(sprintf('%s.isiso', niftiName),isIso);
end
    
end

function [MD, K, b0Est] = computeKurtosis(bShells, yin)
    A = [ones(length(bShells),1), -bShells(:), bShells(:).^2];
    AtA = A'*A;
    y = permute(yin, [4 1 2 3]);
    x = pinv(AtA)*A'*y(:,:); % TODO: could add regularization of some kind?
    szy = size(y);
    x = reshape(x, [3, szy(2:4)]);
    MD = squeeze(x(2,:,:,:));
    K = 6*squeeze(x(3,:,:,:))./MD.^2;
    b0Est = exp(squeeze(x(1,:,:,:)));
end
