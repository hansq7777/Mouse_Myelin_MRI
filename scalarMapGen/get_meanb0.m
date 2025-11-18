function [] = get_meanb0(niftiName, bvalName)
%function to save mean b0 image given a nifti

niftiName = char(niftiName);
bvalName = char(bvalName);

% Load data
im = niftiread(sprintf('%s.nii.gz', niftiName));

bval = dlmread(sprintf('%s.bval', bvalName));
bval = bval(:);

%find b0 shells
im_b0 = im(:,:,:,(bval<100));
im_b0_avg = mean(im_b0, 4);  

%save b0
im_info = niftiinfo(sprintf('%s.nii.gz', niftiName));
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
%
niftiwrite(single(im_b0_avg), sprintf('%s_mean_b0', niftiName), im_info);
[st, sysOut] = system(sprintf('gzip -f %s_mean_b0.nii',niftiName)); if st; error(sysOut); end

end
