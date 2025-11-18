function [im,phca] = ppfpocs(data, dimFFT, dimPartial, niter, removePhase)
% Fill unacquired portion of k-space using projection onto convex sets
%   See https://ece-classes.usc.edu/ee591/library/Pauly-PartialKspace.pdf
%   See also E.D. Lindskog, E.M. Haacke,and W. Lin, J. Magn Reson., 92,126 (1991).
%   
%   Inputs
%     data:         kspace data, with zero-filled unacquired region
%     dimFFT:       number of dimensions to perform fft up to (e.g., 2 uses 2D Fourier transforms)
%                   note: this should only need to ever be 1 if dimPartial == 1
%     dimPartial:   the dimension along which partial Fourier sampling was performed
%     niter:        the number of iterations (default = 10)
%     removePhase:  remove the low frequency phase (experimental; not recommended in routine use)
%
%   Outputs
%     im:           object-domain image
%     phca:         low frequency phase used in iterations
%
% (c) Corey Baron, 2017-22
%

if nargin<4
  niter = 10;
end
if nargin<5
  removePhase = 0;
end

% Get indices for acquired data and symmetric center part
sz = size(data);
testd = abs(data);
for n=1:ndims(data)
  if (n~=dimPartial)
    testd = sum(testd,n);
  end
end
normval = max([testd(1),testd(end)]);
testd = testd/normval;
szRep = sz;
szRep(dimPartial) = 1;
indsAcq = repmat(testd>0.05, szRep);
testd = min(flipdim(testd,dimPartial),testd);
indsSym = repmat(testd>0.05, szRep);
clear testd

% Create phase map
phc = zeros(size(data),class(data));
phc(indsSym) = data(indsSym);
phca = fftnc(phc,dimFFT);
phca = exp(1i*angle(phca));
phc = fftnc(phc,dimFFT,0);
phc = exp(1i*angle(phc));

% Iterate
im = data;
for n=1:niter
  im(indsAcq) = data(indsAcq); % TODO: this could probably be faster if logical indexing isn't used
  im = fftnc(im,dimFFT,0);
  im = abs(im).*phc;
  im = ifftnc(im,dimFFT,0);
end
im(indsAcq) = data(indsAcq);
im = fftnc(im,dimFFT);

if (removePhase)
  im = im.*conj(phca);
end

end
