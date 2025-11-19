function prepareB1RFlocal(name_B1, name_MTon, maskName, outBase)
%PREPAREB1RFLOCAL Convert Bruker B1 map to RFlocal NIfTI for MTsat.
%   name_B1   : path (with or without .nii/.nii.gz) to raw B1 NIfTI
%   name_MTon : path (with or without .nii/.nii.gz) to MTon NIfTI
%   maskName  : optional mask base path (with or without ext), '' if none
%   outBase   : output base path (no extension), writes outBase.nii.gz and outBase.csv
%
%   Steps:
%     - take volume 2 of B1 (Bruker B1Map, radians),
%     - resample to MTon grid if needed,
%     - convert to degrees, divide by nominal FlipAngle from JSON,
%     - clip to a plausible range and save as RFlocal.

if nargin < 4
  error('Usage: prepareB1RFlocal(name_B1, name_MTon, maskName, outBase)');
end

name_B1   = char(name_B1);
name_MTon = char(name_MTon);
maskName  = char(maskName);
outBase   = char(outBase);

% Strip extensions to get base names
name_B1_base   = stripExt(name_B1);
name_MTon_base = stripExt(name_MTon);

% Resolve B1 path
b1_path = '';
if exist([name_B1_base, '.nii.gz'], 'file')
  b1_path = [name_B1_base, '.nii.gz'];
elseif exist([name_B1_base, '.nii'], 'file')
  b1_path = [name_B1_base, '.nii'];
else
  error('B1 NIfTI not found for base "%s"', name_B1_base);
end

% Resolve MTon path
mt_path = '';
if exist([name_MTon_base, '.nii.gz'], 'file')
  mt_path = [name_MTon_base, '.nii.gz'];
elseif exist([name_MTon_base, '.nii'], 'file')
  mt_path = [name_MTon_base, '.nii'];
else
  error('MTon NIfTI not found for base "%s"', name_MTon_base);
end

% Load B1 (expect 4D, use volume 2)
B1_info = niftiinfo(b1_path);
B1_full = single(niftiread(b1_path));
if ndims(B1_full) ~= 4 || size(B1_full, 4) < 2
  error('Expected B1 map with at least 2 volumes (got size %s)', mat2str(size(B1_full)));
end
B1 = squeeze(B1_full(:,:,:,2));  % second volume

% Load MTon
MT_info = niftiinfo(mt_path);
MTon = single(niftiread(mt_path));

% Resample B1 to MTon size if needed
if ~isequal(size(B1), size(MTon))
  B1 = imresize3(B1, size(MTon), 'linear');
end

% Convert radians to degrees
B1_deg = B1 * (180/pi);

% Nominal flip angle from JSON (deg)
flipNom = 1;
jsonPath = [name_B1_base, '.json'];
if exist(jsonPath, 'file')
  J = jsondecode(fileread(jsonPath));
  if isfield(J, 'FlipAngle')
    flipNom = J.FlipAngle;
  end
end
if flipNom <= 0
  flipNom = 1;
end

% Compute RFlocal = local / nominal
RFlocal = B1_deg / flipNom;
RFlocal(~isfinite(RFlocal)) = 0;

% Clip to plausible range
RFlocal(RFlocal < 0.3) = 0.3;
RFlocal(RFlocal > 2.0) = 2.0;

% Optional mask could be used to refine; currently not applied to values
if ~isempty(maskName)
  mask_base = stripExt(maskName);
  mask_path = '';
  if exist([mask_base, '.nii.gz'], 'file')
    mask_path = [mask_base, '.nii.gz'];
  elseif exist([mask_base, '.nii'], 'file')
    mask_path = [mask_base, '.nii'];
  end
  if ~isempty(mask_path)
    % M = niftiread(mask_path) > 0;
    % Optionally, apply mask to RFlocal
  end
end

% Per-slice CSV: all ones (all slices valid)
B1corr_slices = ones(size(RFlocal, 3), 1);
csvPath = sprintf('%s.csv', outBase);
try
  writematrix(B1corr_slices, csvPath);
catch
  dlmwrite(csvPath, B1corr_slices, 'precision', '%.0f');
end

% Write RFlocal NIfTI, compressed
out_info = MT_info;
out_info.MultiplicativeScaling = 1;
out_info.Datatype = 'single';
out_info.BitsPerPixel = 32;
if isfield(out_info, 'raw')
  out_info.raw.datatype = 16;
  out_info.raw.bitpix = 32;
end

niiPath = sprintf('%s.nii', outBase);
niftiwrite(single(RFlocal), niiPath, out_info);
system(sprintf('gzip -f \"%s\"', niiPath));

end

function base = stripExt(pathStr)
%STRIPEXT Remove .nii/.nii.gz from a path if present.
base = char(pathStr);
if endsWith(base, '.nii.gz')
  base = base(1:end-7);
elseif endsWith(base, '.nii')
  base = base(1:end-4);
end
end

