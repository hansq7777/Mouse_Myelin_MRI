function [kspace,phsOut,im] = nyquistAuto(kspace,Nline,kspace_even,is3D)
% 
% Automatically determine linear nyquist ghost correction for EPI with no reference scan
% Uses an entropy cost function.
%   Similar to S. Skare, D. Clayton, R. Newbould, M. Moseley, and R. Bammer, “A
%       fast and robust minimum entropy based non-interactive Nyquist ghost
%       correction algorithm,” in Proc. Int. Soc. Magn. Reson. Med., 2006, p. 2349.
% 
% Inputs
%   kspace:         1st dim = phase encode. 2nd = freq encode. 
%                   Should be in hybrid k-space (already in object domain along dim 2, but k-space along dim 1)
%   Nline:          Number of lines to skip. Should usually be 2 (will not be 2 for multishot or accelerated)
%   kspace_even:    (optional) Even lines of k-space. If provided, "kspace" input is presumed to be the odd lines
%   is3D:           whether it was a 3D acquisition    
%
% Outputs
%   kspaceOut:      k-space of result
%   phsOut:         phase ramps for correction. kspaceOut = phsOut.*kspace_even
%   imOut:          imOut: object-domain representation of result
% 
% (C) Corey Baron, 2017-22
%

% Set default options
if nargin<4 || isempty(is3D)
    is3D = 0;
end
if nargin<3
    kspace_even = [];
end
if ~isempty(kspace_even)
    Nline = 2;
end

if mod(Nline,2)
    warning('Nline should be even')
end
ntries = round(15*sqrt(Nline));

% Switch to double precision
wasSingle = 0;
if isa(kspace,'single')
    kspace = double(kspace);
    kspace_even = double(kspace_even);
    wasSingle = 1;
end

% Scale data for better behaviour of fmincon
scl = max(abs(kspace(:)));
kspace = kspace/scl;
if ~isempty(kspace_even)
    kspace_even = kspace_even/scl;
end
phs = (-size(kspace,2)/2:size(kspace,2)/2-1) / size(kspace,2)*2*pi;

% Do a rough correction of timing to get within one k-space sample of shift
maxVal = -inf;
tryAll = zeros(21,1);
im_a = applyPhc(zeros(1,2*(Nline-1)) ,kspace,phs,kspace_even);
im_a = ifftshift(ifft(ifftshift(im_a,2),[],2),2);
if is3D
    im_a = ifftshift(ifft(ifftshift(im_a,3),[],3),3);
end
scl = max(abs(im_a(:)));
for nv = -10:10
    im_a = applyPhc([zeros(1,Nline-1) zeros(1,Nline/2-1) nv*ones(1,Nline/2)],kspace,phs,kspace_even);
    im_a = ifftshift(ifft(ifftshift(im_a,2),[],2),2);
    if is3D
        im_a = ifftshift(ifft(ifftshift(im_a,3),[],3),3);
    end
    im_a = abs(im_a)/scl;
    
    % Find shift that lines up centroids of all lines near k=0
    [~, mind1] = max(sum(sum(im_a(:,:,:),2),3));
    [~, mind2] = max(sum(sum(im_a(:,:,:),1),3));
    if is3D
        [~, mind3] = max(sum(sum(im_a(:,:,:),1),2));
        mind3 = mind3-6:mind3+5;
    else
        mind3 = 1; %:size(im_a(:,:,:),3);
    end
    cents = sum(im_a(mind1-6:mind1+5,mind2-10:mind2+10,mind3).*(-10:10),2) ./ sum(im_a(mind1-6:mind1+5,mind2-10:mind2+10,mind3),2);
    maxTry = -std(cents(:));

    tryAll(nv+11) = maxTry;
    if maxTry > maxVal
        nvOpt = nv;
        maxVal = maxTry;
    end
end

% Find the precise phase correction
costout = inf;
for nvTry = nvOpt-1:nvOpt+1
    nvAdd = [zeros(1,Nline-1) zeros(1,Nline/2-1) nvTry*ones(1,Nline/2)];
    if is3D
        costfcn = @(phsPar) getCost3D(phsPar + nvAdd,kspace,phs,kspace_even);
    else
        costfcn = @(phsPar) getCost(phsPar + nvAdd,kspace,phs,kspace_even);
    end
    mx = pi;
    par0 = ones(1,2*(Nline-1));

    usePar = true;
    options = optimoptions(@fmincon,'Algorithm', 'interior-point', 'Display', 'none',...
        'DiffMinChange', 0.01, 'DiffMaxChange', 1,... 
        'TypicalX', 1*par0,...
        'TolX', 1e-4,'TolFun',1e-6,...
        'FinDiffType','forward',...
        'UseParallel', usePar);
    problem = createOptimProblem('fmincon',...
        'objective',costfcn,...
        'x0',0*par0,...
        'lb',-mx*par0,'ub',mx*par0,...
        'options', options);
    ms = MultiStart('Display','none',...
        'XTolerance',1e-4,...
        'UseParallel',usePar);
    [parOut_a,costout_a] = run(ms,problem,ntries);
    if costout_a<costout
        costout = costout_a;
        parOut = parOut_a;
        nvOpt_out = nvTry;
    end
end
nvOpt = nvOpt_out;
nvAdd = [zeros(1,Nline-1) zeros(1,Nline/2-1) nvOpt*ones(1,Nline/2)];

% Apply optimal correction
if nvOpt ~= 0
    parOut = parOut+nvAdd;
end
kspace = applyPhc(parOut,kspace,phs,kspace_even)*scl;

% DC phase reduncancy can cause a N/2 shift in the image. Find position
% that minimizes signal near the edges
im1 = fftnc(kspace,1);
im2 = fftshift(im1,1);
E1 = sum(abs(reshape(im1([1,end],:),[],1)));
E2 = sum(abs(reshape(im2([1,end],:),[],1)));
if E2 < E1
    parOut(1:2:end/2) = parOut(1:2:end/2) + pi;
    kspace = applyPhc(parOut,kspace,phs,kspace_even)*scl;
end

% Format output
if wasSingle
    kspace = single(kspace);
end
phsOut = cell(Nline-1,1);
for n=2:Nline
    phsOut{n-1} = exp(1i*(parOut(n-1) + parOut(n-1+Nline-1)*phs));
end
if nargout>2
    im = fftnc(kspace,1);
end

end

%%%% Sub-functions %%%%

function cost = getCost(phsPar,kspace,phs,kspace_even)
    kspace = applyPhc(phsPar,kspace,phs,kspace_even);
    cost = abs(fftnc(kspace,1));
    if ndims(cost)>2
        cost = calcRSOS(cost(:,:,:),3);
    end
    cost = entropy(cost)^2;
end

function cost = getCost3D(phsPar,kspace,phs,kspace_even)
    kspace = applyPhc(phsPar,kspace,phs,kspace_even);
    cost = abs(fftnc(kspace,1));
    cost = entropy(cost(:,:))^2;
end

function kspace = applyPhc(phsPar,kspace,phs,kspace_even)
    % Apply correction using a phase ramp in image domain
    Nline = numel(phsPar)/2 + 1;
    for n=2:Nline
        phsCor = exp(1i*(phsPar(n-1) + phsPar(n-1+Nline-1)*phs));
        if isempty(kspace_even)
            kspace(n:Nline:end,:,:) = kspace(n:Nline:end,:,:).*phsCor;
        else
            kspace_even = kspace_even.*phsCor;
            kspace = kspace + kspace_even;
        end
    end

end
