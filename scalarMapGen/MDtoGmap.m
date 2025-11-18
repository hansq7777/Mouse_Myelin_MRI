function [] = MDtoGmap(varargin)
%Function to generate diffusion dispersion maps
%get a voxel-wise least-square fit of MD to frequency and
%generate a map of the fit factor G
%
%input: filenames for MD maps in order from lowest to highest frequency 
%followed by filename for a mask
%last input is list of frequencies
%filenames should not include extension (.nii.gz)

for n = 1:(nargin-2)
    MDmaps(:,:,:,n) = niftiread(sprintf('%s.nii.gz',varargin{n}));
end

imMask = niftiread(sprintf('%s.nii.gz',varargin{end-1}));
imMask = flip(imMask, 1);
imMask (imMask > 0) = 1;
im_info = niftiinfo(sprintf('%s.nii.gz', varargin{1}));
f = varargin{end};

X = im_info.ImageSize(1);
Y = im_info.ImageSize(2);
Z = im_info.ImageSize(3);

%voxel-wise fit to get G map
for z = 1:Z
    for y = 1:Y
        for x = 1:X
            p(x,y,z,:) = polyfit(sqrt(f), MDmaps(x,y,z,:).*1e+3, 1);
            G(x,y,z) = p(x,y,z,1);
        end
    end
end

%save Gmap
G = single(G);
imMask = single(imMask);
G = G.*imMask;

im_info.Datatype = 'single';
im_info.BitsPerPixel = 32;
im_info.raw.datatype = 16;
im_info.raw.bitpix = 32;

niftiwrite(single(G), sprintf('%s_Gfactor', varargin{1}), im_info, 'Compressed', true);

end












