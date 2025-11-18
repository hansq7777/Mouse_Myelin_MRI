function [] = niftiOrientation(varargin)
%Function to change orientation of given nifti so that cortex is
%in the superior direction

for im = 1:nargin

    niftiName = char(varargin{1,im});

    info = niftiinfo(sprintf('%s.nii.gz',niftiName));
    im = niftiread(sprintf('%s.nii.gz',niftiName));

    im_flip = flip(im,2);

    niftiwrite(im_flip,sprintf('%s',niftiName), info);

    cmd = sprintf('gzip -f %s.nii', niftiName);
    [st, sysOut] = system(cmd); if st; error(sysOut); end

end