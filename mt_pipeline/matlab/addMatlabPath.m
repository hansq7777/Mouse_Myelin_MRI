function addMatlabPath
%ADDMATLABPATH Add MT pipeline MATLAB utilities to the path.
% Keeps scope narrow (only mt_pipeline/matlab) to avoid pulling in .git, etc.

% Delete any parallel pools, because the workers won't get the new path otherwise.
if exist('gcp', 'file')
  poolobj = gcp('nocreate');
  delete(poolobj);
end

matlabDir = fileparts(mfilename('fullpath')); % .../mt_pipeline/matlab
addpath(genpath(matlabDir));

clear matlabDir poolobj
end
