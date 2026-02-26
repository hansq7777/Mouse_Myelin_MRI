function [t1wPath, t2wPath, ratioPath] = nii2t1wt2wr(t1Path, t2Path, unusedArg, maskPath, useGauss, matchIntensity) %#ok<INUSD>
%NII2T1WT2WR Generate T1w, T2w, and ratio volumes from input NIfTI files.
%   nii2t1wt2wr(t1Path, t2Path, ~, maskPath, useGauss, matchIntensity)
%     t1Path          : path to T1-weighted image (.nii or .nii.gz)
%     t2Path          : path to T2-weighted / RAREvfl image (.nii or .nii.gz)
%     maskPath        : optional binary mask (same grid as T2); '' to skip
%     useGauss        : 1 to apply mild Gaussian smoothing (default sigma=1)
%     matchIntensity  : 1 to match T1 intensity scale to T2 within mask
%
%   Outputs are written next to the T2 input using suffixes:
%     *_t1w.nii.gz, *_t2w.nii.gz, *_t1wT2wRatio.nii.gz
%
%   The function resamples the T1 data to the T2 grid if necessary, performs
%   optional smoothing, rescales intensities to [0,1] based on 1st-99th
%   percentiles within the mask, and computes the simple ratio T1w./T2w.

if nargin < 4
  maskPath = '';
end
if nargin < 5 || isempty(useGauss)
  useGauss = 1;
end
if nargin < 6 || isempty(matchIntensity)
  matchIntensity = 1;
end

t1Path = char(t1Path);
t2Path = char(t2Path);
maskPath = char(maskPath);

t2info = niftiinfo(resolve_existing_nifti(t2Path));
t2vol = single(niftiread(t2info));
t1info = niftiinfo(resolve_existing_nifti(t1Path));
t1vol = single(niftiread(t1info));

if ~isequal(size(t1vol), size(t2vol))
  t1vol = resize_like(t1vol, size(t2vol));
end

mask = true(size(t2vol));
if ~isempty(strtrim(maskPath))
  maskData = load_mask(maskPath, size(t2vol));
  if ~isempty(maskData)
    mask = maskData > 0;
  end
end

if useGauss
  t1vol = smooth_volume(t1vol);
  t2vol = smooth_volume(t2vol);
end

if matchIntensity
  t1_scale = robust_scale(t1vol, mask);
  t2_scale = robust_scale(t2vol, mask);
  if t1_scale.width > 0 && t2_scale.width > 0
    gain = t2_scale.median / max(t1_scale.median, eps);
    t1vol = t1vol * gain;
  end
end

t1w = normalize_volume(t1vol, mask);
t2w = normalize_volume(t2vol, mask);
ratio = zeros(size(t1w), 'single');
ratio(mask) = t1w(mask) ./ max(t2w(mask), eps);
ratio(~isfinite(ratio)) = 0;
ratio = min(ratio, 5);

[outDir, baseName] = strip_ext(t2Path);
baseOut = fullfile(outDir, baseName);
t1wPath = sprintf('%s_t1w.nii.gz', baseOut);
t2wPath = sprintf('%s_t2w.nii.gz', baseOut);
ratioPath = sprintf('%s_t1wT2wRatio.nii.gz', baseOut);

write_nifti(t1wPath, t2info, t1w);
write_nifti(t2wPath, t2info, t2w);
write_nifti(ratioPath, t2info, ratio);

fprintf('nii2t1wt2wr: wrote %s, %s, %s\n', t1wPath, t2wPath, ratioPath);

end

% -------------------------------------------------------------------------
function normVol = normalize_volume(vol, mask)
data = double(vol);
vox = data(mask);
vox = vox(isfinite(vox));
if isempty(vox)
  vox = data(isfinite(data));
end
if isempty(vox)
  normVol = zeros(size(vol), 'single');
  return;
end
p1 = prctile(vox, 1);
p99 = prctile(vox, 99);
scale = max(p99 - p1, eps);
normVol = single((data - p1) / scale);
normVol(normVol < 0) = 0;
normVol(~isfinite(normVol)) = 0;
end

function stats = robust_scale(vol, mask)
vals = double(vol(mask));
vals = vals(isfinite(vals));
if isempty(vals)
  stats = struct('median', 0, 'width', 0);
  return;
end
stats.median = median(vals);
stats.width = prctile(vals, 90) - prctile(vals, 10);
end

function out = smooth_volume(vol)
try
  out = imgaussfilt3(vol, 1);
catch
  out = smooth3(vol, 'gaussian', [3 3 3], 0.65);
end
end

function data = load_mask(pathStr, targetSize)
pathResolved = resolve_existing_nifti(pathStr);
info = niftiinfo(pathResolved);
mask = niftiread(info);
mask = squeeze(mask);
if ~isequal(size(mask), targetSize)
  mask = resize_like(single(mask), targetSize) > 0.5;
end
data = mask;
end

function arr = resize_like(arr, newSize)
if isequal(size(arr), newSize)
  return;
end
if exist('imresize3', 'file') == 2
  arr = imresize3(arr, newSize, 'linear');
  arr = single(arr);
  return;
end
[X, Y, Z] = ndgrid(1:size(arr,1), 1:size(arr,2), 1:size(arr,3));
[Xq, Yq, Zq] = ndgrid( ...
  linspace(1, size(arr,1), newSize(1)), ...
  linspace(1, size(arr,2), newSize(2)), ...
  linspace(1, size(arr,3), newSize(3)));
arr = interp3(Y, X, Z, double(arr), Yq, Xq, Zq, 'linear', 0);
arr = single(arr);
end

function write_nifti(pathOut, templateInfo, volume)
info = templateInfo;
info.Datatype = 'single';
info.BitsPerPixel = 32;
info.MultiplicativeScaling = 1;
if isfield(info, 'raw')
  info.raw.datatype = 16;
  info.raw.bitpix = 32;
end

if endsWithi(pathOut, '.nii.gz')
  tmpPath = pathOut(1:end-3);
else
  tmpPath = pathOut;
end
niftiwrite(volume, tmpPath, info, 'Compressed', false);
gzip(tmpPath);
delete(tmpPath);
end

function pathResolved = resolve_existing_nifti(pathStr)
candidates = candidate_paths(pathStr);
for ii = 1:numel(candidates)
  if exist(candidates{ii}, 'file') == 2
    pathResolved = candidates{ii};
    return;
  end
end
error('File not found for %s', pathStr);
end

function [folder, baseName] = strip_ext(pathStr)
resolved = resolve_existing_nifti(pathStr);
[folder, name, ext] = fileparts(resolved);
if strcmpi(ext, '.gz')
  [~, name, ~] = fileparts(name);
end
baseName = name;
end

function c = candidate_paths(pathStr)
p = char(pathStr);
c = {p};
if ~endsWithi(p, '.nii') && ~endsWithi(p, '.nii.gz')
  c{end+1} = [p, '.nii']; %#ok<AGROW>
  c{end+1} = [p, '.nii.gz']; %#ok<AGROW>
elseif endsWithi(p, '.nii')
  c{end+1} = [p, '.nii.gz']; %#ok<AGROW>
elseif endsWithi(p, '.nii.gz')
  c{end+1} = p(1:end-3); %#ok<AGROW>
end
end

function tf = endsWithi(str, pat)
tf = endsWith(lower(str), lower(pat));
end
