function addMatlabPath
% Note: we avoid genpath on top repo dir as it will add the whole .git
% folder and its subdirectories too.

% Delete any parallel pools, because the workers won't get the new path otherwise
if(exist('gcp'))
  poolobj = gcp('nocreate');
  delete(poolobj);
end

topPath = mfilename('fullpath');
topPath = topPath(1:end-length(mfilename));

% Add paths
allPaths = [topPath(1:end-1),';',...
  genpath([topPath, 'niiCombDiffAve']),...
  genpath([topPath, 'scalarMapGen'])
];
addpath(allPaths);

clear allPath topPath

end
