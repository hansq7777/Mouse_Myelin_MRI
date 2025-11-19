function recenterNifti(inPath, outPath)
%RECENTERNIFTI Reset NIfTI origin to volume center and save.
%   recenterNifti('in.nii[.gz]', 'out.nii[.gz]')

info = niftiinfo(inPath);
V = niftiread(info);

if isfield(info, 'Transform') && isfield(info.Transform, 'T')
    T = info.Transform.T;
    T(4,1:3) = 0;
    info.Transform.T = T;
end

% Infer compression from extension
compressed = endsWith(outPath, '.gz');
niftiwrite(V, outPath, info, 'Compressed', compressed);
end
