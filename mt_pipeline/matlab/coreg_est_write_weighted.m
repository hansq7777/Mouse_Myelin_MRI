function outPath = coreg_est_write_weighted(sourcePath, refPath, maskPath, prefix, interp)
%COREG_EST_WRITE_WEIGHTED Run SPM coregistration with optional weighting.
%   coreg_est_write_weighted(source, ref, mask, prefix, interp)
%     source : moving image (MTon, etc.), accepts .nii or .nii.gz
%     ref    : fixed image (MToff / PDw)
%     mask   : optional weighting image ('' to skip). If provided it is
%              assigned to wref so SPM focuses on in-mask voxels.
%     prefix : prefix for resliced outputs (default: 'r')
%     interp : interpolation order (0=nearest, 1=trilinear, 4=spline, ...)
%
%   The function handles .nii.gz inputs by copying them to a temp folder,
%   running SPM's spm.spatial.coreg.estwrite job, and writing the result
%   back next to the source image (matching the original compression).

if nargin < 3
  maskPath = '';
end
if nargin < 4 || isempty(prefix)
  prefix = 'r';
end
if nargin < 5 || isempty(interp)
  interp = 4;
end

sourcePath = char(sourcePath);
refPath = char(refPath);
maskPath = char(maskPath);
prefix = char(prefix);

srcInfo = prepare_input(sourcePath, 'source');
refInfo = prepare_input(refPath, 'ref');
if ~isempty(strtrim(maskPath))
  maskInfo = prepare_input(maskPath, 'mask');
else
  maskInfo = empty_info();
end

cleanupObj = onCleanup(@() cleanup_temp_dirs([srcInfo, refInfo, maskInfo])); %#ok<NASGU>

if exist('spm', 'file') ~= 2 || exist('spm_jobman', 'file') ~= 2
  error('coreg_est_write_weighted:MissingSPM', ...
    ['SPM not found on the MATLAB path. Run addpath(''/path/to/spm'') ', ...
     'before calling this function.']);
end

spm('defaults', 'fmri');
spm_jobman('initcfg');

matlabbatch = {};
matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {refInfo.run_path}; %#ok<*AGROW>
if ~isempty(maskInfo.run_path)
  matlabbatch{1}.spm.spatial.coreg.estwrite.wref = {maskInfo.run_path};
else
  matlabbatch{1}.spm.spatial.coreg.estwrite.wref = {''};
end
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {srcInfo.run_path};
matlabbatch{1}.spm.spatial.coreg.estwrite.other = {''};
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = ...
  [0.0200 0.0200 0.0200 0.0010 0.0010 0.0010 0.0100 0.0100 0.0100 0.0010 0.0010 0.0010];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = interp;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = prefix;

spm_jobman('run', matlabbatch);

tmpOutput = prefixed_path(srcInfo.run_path, prefix);
if ~exist(tmpOutput, 'file')
  error('coreg_est_write_weighted:MissingOutput', ...
    'Expected coreg output not found: %s', tmpOutput);
end

destBase = fullfile(srcInfo.output_dir, sprintf('%s%s', prefix, srcInfo.base_name));
destNii = sprintf('%s.nii', destBase);
if ~strcmpi(tmpOutput, destNii)
  if exist(destNii, 'file')
    delete(destNii);
  end
  copyfile(tmpOutput, destNii);
end

if srcInfo.output_gz
  gzip(destNii);
  delete(destNii);
  destPath = sprintf('%s.nii.gz', destBase);
else
  destPath = destNii;
end

if nargout > 0
  outPath = destPath;
end
fprintf('coreg_est_write_weighted: wrote %s\n', destPath);

end

% -------------------------------------------------------------------------
function info = prepare_input(pathStr, label)
info = empty_info();
if isempty(strtrim(pathStr))
  error('Empty path provided for %s', label);
end

resolved = resolve_existing_nifti(char(pathStr));
[info.output_dir, fname, ext] = fileparts(resolved);
info.base_name = fname;
info.output_gz = false;

if endsWithi(ext, '.gz')
  info.output_gz = true;
  [~, inner_name, inner_ext] = fileparts(fname);
  zipped_name = sprintf('%s%s%s', inner_name, inner_ext, ext);
  info.base_name = inner_name;
  temp_dir = tempname;
  mkdir(temp_dir);
  tmp_gz = fullfile(temp_dir, zipped_name);
  copyfile(resolved, tmp_gz);
  gunzip(tmp_gz, temp_dir);
  delete(tmp_gz);
  info.run_path = fullfile(temp_dir, sprintf('%s%s', inner_name, inner_ext));
  info.cleanup_dir = temp_dir;
else
  info.run_path = resolved;
  info.cleanup_dir = '';
end
info.original = resolved;
[~, info.run_name, info.run_ext] = fileparts(info.run_path);
end

function info = empty_info()
info = struct('original', '', 'run_path', '', 'cleanup_dir', '', ...
  'output_dir', '', 'base_name', '', 'run_name', '', 'run_ext', '', ...
  'output_gz', false);
end

function resolved = resolve_existing_nifti(pathStr)
p = char(pathStr);
candidates = {p};
if ~endsWithi(p, '.nii') && ~endsWithi(p, '.nii.gz')
  candidates{end+1} = [p, '.nii']; %#ok<AGROW>
  candidates{end+1} = [p, '.nii.gz']; %#ok<AGROW>
elseif endsWithi(p, '.nii')
  candidates{end+1} = [p, '.gz']; %#ok<AGROW>
elseif endsWithi(p, '.nii.gz')
  candidates{end+1} = p(1:end-3); %#ok<AGROW>
end

for ii = 1:numel(candidates)
  if exist(candidates{ii}, 'file') == 2
    resolved = candidates{ii};
    return;
  end
end
error('File not found (tried %s)', strjoin(candidates, ', '));
end

function tf = endsWithi(str, pat)
tf = endsWith(lower(str), lower(pat));
end

function cleanup_temp_dirs(infos)
for idx = 1:numel(infos)
  if isfield(infos(idx), 'cleanup_dir') && ~isempty(infos(idx).cleanup_dir)
    if exist(infos(idx).cleanup_dir, 'dir')
      rmdir(infos(idx).cleanup_dir, 's');
    end
  end
end
end

function outPath = prefixed_path(runPath, prefix)
[runDir, runName, runExt] = fileparts(runPath);
outPath = fullfile(runDir, sprintf('%s%s%s', prefix, runName, runExt));
end
