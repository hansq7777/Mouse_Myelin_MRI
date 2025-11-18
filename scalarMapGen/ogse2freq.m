function [] = ogse2freq(niftiName, maskName, bvalName, freq, directions)
%Function to split OGSE nifti into niftis containing each freq + all b0s, and split bval and bvec files for each freq
%Filenames must not include extensions (.nii.gz is assumed for nifti, and .bval, .bvec, .json for support files)

%Assumes: OGSE data has ben collected with Shape-Dir-Amp option from
%Bruker scanner, so that each shape is looped through first for each direction

niftiName = char(niftiName);
maskName = char(maskName);
bvalName = char(bvalName);

% Load data
im = niftiread(sprintf('%s.nii.gz', niftiName));
slices = size(im,3);
vol = size(im,4);
shapes = length(freq);

bval = dlmread(sprintf('%s.bval', bvalName));
bvec = dlmread(sprintf('%s.bvec', bvalName));
bval = bval(:);

%find b-shells - bhigh (with diffusion weighting) and b0 
bval_bhigh = bval(bval > 300);
bvec_bhigh = bvec(:,bval > 300);
im_b0 = im(:,:,:,(bval<100));
im_b0_avg = mean(im_b0, 4);
im_bhigh = im(:,:,:,(bval > 300));
b0_ind = find(bval < 100);          

for sh = 1:shapes
    counter = sh;               %counter will go to the correct image volume for a certain direction
    for d = 1:directions
        sig_dir(:,:,:,d) = im_bhigh(:,:,:,counter);
        bval_dir(d) = bval_bhigh(counter);
        bvec_dir(:,d) = bvec_bhigh(:,counter);
        counter = counter + shapes;
    end
    %add on all b0s
    im_freq(:,:,:,:,sh) = cat(4,im_b0, sig_dir);
    bval_dir = bval_dir(:);
    bval_freq(:,sh) = cat(1, bval(bval < 100), bval_dir);
    bvec_freq(:,:,sh) = cat(2, bvec(:,bval<100), bvec_dir);
end

%save niftis
im_info = niftiinfo(sprintf('%s.nii.gz', niftiName));
im_info.ImageSize(4) = directions + size(im_b0,4);
%
im_info.Datatype = 'single';
im_info.BitsPerPixel = 32;
im_info.raw.datatype = 16;
im_info.raw.bitpix = 32;
% niftiwrite doesn't support names with periods
niftiName(niftiName == '.') = '_';
%
bval_freq = bval_freq';
for sh = 1:shapes
    im_save = im_freq(:,:,:,:,sh);
    if freq(sh) < 10
        p = '00';
    elseif freq(sh) < 100
        p = '0';
    else
        p = '';
    end
    niftiwrite(single(im_save), sprintf('%s_f%s%s', niftiName, p, num2str(freq(sh))), im_info);
    [st, sysOut] = system(sprintf('gzip -f %s_f%s%s.nii',niftiName, p, num2str(freq(sh)))); if st; error(sysOut); end
    dlmwrite(sprintf('%s_f%s%s.bval', niftiName, p, num2str(freq(sh))), bval_freq(sh,:), 'delimiter', ' ');
    dlmwrite(sprintf('%s_f%s%s.bvec', niftiName, p, num2str(freq(sh))), bvec_freq(:,:,sh), 'delimiter', ' ');
end

end

