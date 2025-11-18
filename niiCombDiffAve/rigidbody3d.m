function [p_f,T,costout,imgout,exval,output] = rigidbody3d(img_ref,img_in,opt,useGPU)
%   Performs rigid body shifts and rotations either via registration with
%   reference image, or through rotation and shift specification. Can use
%   autocorrelation or mean squared difference between volumes, or a joint-histogram based cost
%   funtion [MRM 51:103 (2004)]. The latter is
%   slower, but less affected by differences in contrast between the
%   volumes. For similar contrast mean squared diff seems to have the
%   best performance.
%
%   Usage: [p_f,T,costout,imgout,exval,output] =
%   rigidbody3d(img_ref,img_in,opt)
%   (c) Corey Baron, 2015 - 22
%
%   Input:
%   img_ref: The volume to be used as a reference.
%   img_in: The volume to be motion corrected wrt img_ref. Complex data
%           supported, however only abs(img_in) used during iteration.
%           Final output retains image phase.
%   opt:    struct containing options
%   useGPU: set to 0 to disable GPU (default = 1)
%
%   Possible opt fields:
%   immask: mask of region of interest used for cost function
%   voxsz:  Voxel size. Units do not matter, but relative voxel dimensions
%           are important.
%   p_in:   transformation parameters should the user just want to specify
%           them manually. A 6 element vector: p_in(1:3) = shifts in
%           pixels [dim1 dim2 dim3], p_in(4:6) = rotations in degrees,
%           about [dim1 dim2 dim3]
%   costfcn_type: the cost function used. 1 for autocorrelation,
%           2 for joint histogram (slower, but performs
%           okay with different contrast [MRM 51:103 (2004)]), 3 for mean
%           squared difference. Default = 3.
%   cntr:   the voxel location used for center of rotation. Default is
%           size(img_in)/2.
%   maxshfts_rot: largest shift and rotation allowed (voxels
%                 and degrees) (just a single number)
%   interptype: 'fft' for fourier transform (recommended for 3d data)
%               'spatial' for spatial domain interpolation (recommended for 3d motion tracking of a
%                   stack of 2d slice acquired data b/c fft causes ringing). Does not use image phase.
%   mocodims:  'aff'      for 12 parameter affine
%              '3D'       for 3 translations and 3 rotations (default)
%                         NB: Rotations use Tait-Bryan formalism (NOT Euler!)
%              '3Dshift'  for 3 translations and 0 rotations
%              '2D'       for 2 translations and 1 rotation on first two
%                         dims of input. Whole volume is used together if
%                         number of dims > 2.
%              '1D'
%
%
%   Output:
%   imgout: The motion corrected volume.
%   p_f:    The shifts and rotations. Same format as p_in.
%   T:      transformation matrix to go from img_in to img_ref. Last
%           column is the shifts
%   exval,output: outputs of fmincon function.

% TODO: this needs unit tests. One should test consistency between fft and
%   spatial interptype's
% TODO: change method for getting options in. Should be a struct (eval
% method is messy)
% TODO: refactoring tasks: 
%   1. do not use embedded functions (messy!)
%   2. simplify inputs
%   3. reorder outputs. im and p should be first two (im first!)
%   4. make this a class, so that basic matrices can be stored for use in
%       other iterative functions.


% Set default parameter defaults
optDisplay = 'none';
doabs = 0;
voxsz = [1 1 1];
p_in = [];
costfcn_type = 3;
cntr = [];
immask = [];
interptype = 'fft';
mocodims = '3D';
% Solver options
maxshfts_rot = 5;   % voxels for shifts, and degrees for rotations
TolFun_val = 1e-8;
Tolx_val = 1e-6;
typx = 0.5;         % Typical shifts (in voxels) or rotation (in
                    % degrees)
doreduce = 0;  % This greatly speeds things up when a mask is
               % used (immask), but is potentially less accurate (xform
               % can introduce ringing near the new boundaries).

if nargin<4
    useGPU = 1;
end

% Load in supplied parameters
if(nargin>2 && ~isempty(opt))
    fnames = fieldnames(opt);
    for f=1:length(fnames)
        fname = fnames{f};
        val = getfield(opt, fname);
        eval(sprintf('%s = val;', fname));
    end
end

% Force interptype for affine
if strcmp(mocodims, 'aff')
  interptype = 'aff';
end

% Choose interpolating function
switch interptype
    case 'aff'
        costfcn = @(pvals)rb_xform_affine(pvals);
    case 'fft'
        costfcn = @(pvals)rb_xform(pvals);
    case 'spatial'
        costfcn = @(pvals)rb_xform_spatial(pvals);
        doabs = 1;
        if strcmp(mocodims,'aff')
            mocodims = '3D';
            disp('affine not supported for interptype = ''spatial'' ')
        end
end

% Get size of volume array
sz = size(img_in);
sz = [sz,1,1,1];
if prod(sz(2:end)) == 1
  mocodims = '1D';
elseif prod(sz(3:end)) == 1
  mocodims = '2D';
end

% Check if data set is too big (looping through volumes should be done
% outside this function)
if prod(sz(4:end)) > 1
    error('Matrix dimension larger than 3.')
end

% Check if input data is real (if so, output is forced to be real later)
real_in_flag = 0;
if isreal(img_in)
    real_in_flag = 1;
    if ~isreal(img_ref)
        img_ref = abs(img_ref);
    end
end

% Check for single precision input
issingle = 0;
if strcmp(class(img_ref),'single')
    img_ref = double(img_ref);
    img_in = double(img_in);
    issingle = 1;
end

% Initialize output variables
imgout = [];
p_f = [];
exval = [];
output = [];
costout = [];
T = [];

% If odd number in a dimension, pad with zeros to make it even.
if any(mod(sz,2) == 1)
    if mod(sz(1),2) == 1
        img_in = cat(1,img_in,zeros(1,sz(2),sz(3)));
        img_ref = cat(1,img_ref,zeros(1,sz(2),sz(3)));
    end
    if (mod(sz(2),2) == 1) && (sz(2)~=1)
        img_in = cat(2,img_in,zeros(sz(1),1,sz(3)));
        img_ref = cat(2,img_ref,zeros(sz(1),1,sz(3)));
    end
    if (mod(sz(3),2) == 1) && (sz(3)~=1)
        img_in = cat(3,img_in,zeros(sz(1),sz(2),1));
        img_ref = cat(3,img_ref,zeros(sz(1),sz(2),1));
    end
    SZ = sz;
    sz = [size(img_in), 1, 1, 1];
end

% If no mask supplied, make mask prevent edge voxels from getting used
if isempty(immask)
  immask = false(size(img_in));
  immask(maxshfts_rot+1:end-maxshfts_rot,...
         maxshfts_rot+1:end-maxshfts_rot,...
         maxshfts_rot+1:end-maxshfts_rot) = true;
end

% Save version of input that we will modify
if doabs
    img_ref = abs(img_ref);
    img_in = abs(img_in);
end
img_ref_a = img_ref;
img_curr_a = img_in;

% Perform a blur for spatial case where linear interpolation is used during iterations. 
if strcmp(interptype, 'spatial')
    lpfFact = 0.3;
    img_ref_a = lpfImage(img_ref_a,'gauss',lpfFact);
    img_curr_a = lpfImage(img_curr_a,'gauss',lpfFact);
end

% Define center of rotation
if isempty(cntr)
    cntr = floor(sz/2) + 1;
end
cntr_a = cntr;

% Reduce volume size to only include region with mask buffered by max shifts
if ~isempty(immask) && doreduce
    % Find reduced size
    tmp = sum(sum(immask,2),3);
    rng(:,1) = [find(tmp>0,1,'first')-maxshfts_rot,...
                find(tmp>0,1,'last')+maxshfts_rot];
    tmp = sum(sum(immask,1),3);
    rng(:,2) = [find(tmp>0,1,'first')-maxshfts_rot,...
                find(tmp>0,1,'last')+maxshfts_rot];
    tmp = sum(sum(immask,1),2);
    rng(:,3) = [find(tmp>0,1,'first')-maxshfts_rot,...
                find(tmp>0,1,'last')+maxshfts_rot];
    % Check bounds
    rng(1,:) = max(rng(1,:),1);
    rng(2,:) = min(rng(2,:),sz);
    % Force even number
    num = mod(diff(rng,1)+1,2);
    for ii=1:3
        if num(ii)
            if rng(1,ii) > 1
               rng(1,ii) = rng(1,ii)-1;
            else
               rng(2,ii) = rng(2,ii)+1;
            end
        end
    end
    % Reduce volume size
    sz0 = sz;
    img_ref_a = img_ref_a(rng(1,1):rng(2,1),rng(1,2):rng(2,2),rng(1,3):rng(2,3));
    img_curr_a = img_curr_a(rng(1,1):rng(2,1),rng(1,2):rng(2,2),rng(1,3):rng(2,3));
    immask = immask(rng(1,1):rng(2,1),rng(1,2):rng(2,2),rng(1,3):rng(2,3));
    sz = size(img_ref_a);
    cntr_a = cntr_a - rng(1,:) + 1;
end


% Create basic shear matrices
if useGPU
    useclass = 'gpuArray';
else
    useclass = class(img_in);
end
if isempty(p_in)
    [basic_2wrt3, basic_3wrt2, basic_1wrt3, basic_3wrt1, basic_1wrt2, basic_2wrt1,...
        basic_shf_1, basic_shf_2, basic_shf_3,basic_scl_1,basic_scl_2,basic_scl_3] = ...
        createbasics(sz,cntr_a,useclass);
end

% Set initial guesses for fmincon
switch mocodims
    case '1D'
         npar = 1;
         costfcn = @(pvals)rb_1d(pvals);
         interptype = '1D';
    case {'3Dshift','2D'}
         npar = 3;
    case '3D'
         npar = 6;
    otherwise
         npar = 12;
end
minsh = -maxshfts_rot.*ones(1,npar);
maxsh = maxshfts_rot.*ones(1,npar);
p0 = zeros(1,npar);
typx_all = typx*ones(1,npar);

% Set up shared variables for nested costfcn
sub1 = [];
sub2 = [];
sub3 = [];
sub4 = [];
sub5 = [];
pmap1 = [];
p_a = NaN(1,12);

% Doing the first fft here saves 1 fft every iteration
switch interptype
  case '1D'
    img_curr_a = fft(img_curr_a,[],1);
  case 'fft'
    img_curr_a = fft(img_curr_a,[],2);
  case 'spatial'
    img_curr_a = abs(img_curr_a);
end

% Normalize images so that costfcn ~ 1 for starting case
fact = costfcn(p0);
img_curr_a = img_curr_a/fact;
img_ref_a = img_ref_a/fact;

% Perform search for motion parameters
if isempty(p_in)
    % Perform fmincon.
    [p_f,costout,exval,output] = fmincon(costfcn, p0,...
        [],[],[],[],minsh,maxsh,[],...
        optimoptions('fmincon',...
        'OptimalityTolerance',TolFun_val,...
        'StepTolerance',Tolx_val,...
        'FiniteDifferenceStepSize', eps^0.25,...  % default sqrt(eps). Increases convergence, because very small step sizes only have a small cost change for this type of problem.
        'display',optDisplay));
else
    % If transformation parameters provided, simply apply them.
    p_f = p_in;
end

% Create output
if nargout > 2
    % Apply transformations to original input image
    % Revert cropped/smoothed images back
    img_ref_a = img_ref;
    img_curr_a = img_in;
    sz = size(img_ref);
    immask = ones(sz);
    switch interptype
      case {'fft', '1D'}
        if strcmp(interptype,'1D')
            img_curr_a = fft(img_curr_a,[],1);
        else
            img_curr_a = fft(img_curr_a,[],2);
        end
        [basic_2wrt3, basic_3wrt2, basic_1wrt3, basic_3wrt1, basic_1wrt2, basic_2wrt1,...
           basic_shf_1, basic_shf_2, basic_shf_3,basic_scl_1,basic_scl_2,basic_scl_3] = ...
           createbasics(sz,cntr,useclass);
      case 'spatial'
        img_curr_a = abs(img_curr_a);
    end
    % Reset some shared variables
    sub1 = [];
    sub2 = [];
    sub3 = [];
    sub4 = [];
    sub5 = [];
    pmap1 = [];
    p_a = NaN(1,12);
    % Do a final transformation
    img_curr = [];
    if strcmp(interptype,'spatial')
        useGPU = 0; % Want to use GPU for iterations, but GPU only supports linear interpolation for 'spatial' interptype. So, we use makima for the final interpolation after finding the params with linear.
    end
    costfcn(p_f);
    imgout = img_curr;
    
    if isa(imgout, 'gpuArray')
        imgout = gather(imgout);
    end

    % If padded with zeros to get even matrix dimensions, remove now.
    if exist('SZ','var')
        imgout = imgout(1:SZ(1),1:SZ(2),1:SZ(3));
    end

    % Make sure output is real for real inputs
    if 0% real_in_flag && ~isreal(imgout)
        imgout = abs(imgout);
    end

    % Check for single precision input
    if issingle
      imgout = single(imgout);
    end
end

% Get transformation matrix
if strcmp(mocodims,'aff')
  T1 = [(1-p_f(4)/100), tan(-p_f(8)*pi/180), tan(-p_f(7)*pi/180); 0, 1, 0; 0, 0, 1];
  T2 = [1, 0, 0; tan(-p_f(9)*pi/180), (1-p_f(5)/100), tan(-p_f(10)*pi/180); 0, 0, 1];
  T3 = [1, 0, 0; 0, 1, 0; tan(-p_f(11)*pi/180), tan(-p_f(12)*pi/180), (1-p_f(6)/100)];
  T = T3*T2*T1;
elseif strcmp(mocodims,'3D')
  T = rotTaitBryan(-p_f(4)*pi/180,-p_f(5)*pi/180,-p_f(6)*pi/180);
else
  T = zeros(3,3);
end
if length(p_f) < 3
  p_f = [p_f, zeros(1,3-length(p_f))];
end
T = cat(2,T,p_f(1:3).');

% END main function

%% nested subfunction: for trival 1D case
function Q = rb_1d(p_in)
% p(1:3) are translations (pixels); p(4:6) are rotations (degrees)
% voxsz = dimensions of voxels. Important for rotation.

% Apply shift
pmap = basic_shf_1 * p_in;
img_curr = img_curr_a .* exp(1i*pmap);
img_curr = ifft(img_curr,[],1);

if doabs
    img_curr = abs(img_curr);
end

% Calculate cost function
Q = calccost(costfcn_type,img_curr,img_ref_a,immask);
end


%% nested subfunction: for 3D iteration where all rotations are computed
% before all translations
function Q = rb_xform(p_in)
% p(1:3) are translations (pixels); p(4:6) are rotations (degrees)
% voxsz = dimensions of voxels. Important for rotation.

% Set up motion values depending on mocodims
p = fixp(p_in,mocodims);

% Determine shift of outermost pixel for rotation.
rots = p(4:6)*pi/180;
p_shf_1(1) = - (sz(3) / 2) * tan( rots(1)/2 ) * voxsz(3)/voxsz(2);
p_shf_1(2) =  (sz(2) / 2) * sin( rots(1) ) * voxsz(2)/voxsz(3);
p_shf_2(1) = (sz(3) / 2) * sin( rots(2) ) * voxsz(3)/voxsz(1);
p_shf_2(2) =  - (sz(1) / 2) * tan( rots(2)/2 ) * voxsz(1)/voxsz(3);
p_shf_3(1) = - (sz(2) / 2) * tan( rots(3)/2 ) * voxsz(2)/voxsz(1);
p_shf_3(2) = (sz(1) / 2) * sin( rots(3) ) * voxsz(1)/voxsz(2);

% Apply 1D ffts for rotations and shifts.

% Note that some rotations and shifts are only performed again if they are
% different from the last iteration (since not all of p are changed
% every iteration) and are non-zero.

if useGPU && ~isa(img_curr_a, 'gpuArray')
    img_curr = gpuArray(img_curr_a);
else
    img_curr = img_curr_a;
end

% Rotation 1
if isequal(p(4), p_a(4))
 img_curr = sub1;
elseif p(4)~=0
  pmap = basic_2wrt3 * p_shf_1(1);  % half of shear 3a
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],2);
  img_curr = fft(img_curr,[],3);
  pmap = basic_3wrt2 * p_shf_1(2);  % shear 3b
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],3);
  pmap = basic_2wrt3 * p_shf_1(1);  % half of shear 3a
  img_curr = fft(img_curr,[],2);
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],2);
  sub1 = img_curr;
else
  img_curr = ifft(img_curr,[],2);
  sub1 = img_curr;
end
% Rotation 2
if isequal(p(4), p_a(4)) && isequal(p(5), p_a(5))
  img_curr = sub2;
elseif p(5)~=0
  img_curr = fft(img_curr,[],3);
  pmap = basic_3wrt1 * p_shf_2(2);  % half of shear 2b
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],3);
  img_curr = fft(img_curr,[],1);
  pmap = basic_1wrt3 * p_shf_2(1);  % shear 2a
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],1);
  pmap = basic_3wrt1 * p_shf_2(2);  % half of shear 2b
  img_curr = fft(img_curr,[],3);
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],3);
  sub2 = img_curr;
else
  sub2 = img_curr;
end
% Rotation 3
if isequal(p(4), p_a(4)) && isequal(p(5), p_a(5)) && isequal(p(6), p_a(6))
  img_curr = sub3;
elseif p(6)~=0
  img_curr = fft(img_curr,[],1);
  pmap = basic_1wrt2 * p_shf_3(1);  % half of shear 1a
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],1);
  img_curr = fft(img_curr,[],2);
  pmap = basic_2wrt1 * p_shf_3(2);  % shear 1b
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],2);
  img_curr = fft(img_curr,[],1);
  pmap = basic_1wrt2 * p_shf_3(1);  % half of shear 1a
  img_curr = img_curr .* exp(1i*pmap);
  sub3 = img_curr;
else
  img_curr = fft(img_curr,[],1);
  sub3 = img_curr;
end

% Shifts
if any(p(1:3) ~= 0)
  img_curr = fft(fft(img_curr,[],2),[],3);
  pmap = basic_shf_1 * p(1) + basic_shf_2 * p(2) + basic_shf_3 * p(3);
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(ifft(ifft(img_curr,[],1),[],2),[],3);
else
  img_curr = ifft(img_curr,[],1);
end

p_a = p;

if doabs
    img_curr = abs(img_curr);
end

% Calculate cost function
Q = calccost(costfcn_type,img_curr,img_ref_a,immask);

end

%% nested subfunction for affine
function Q = rb_xform_affine(p_in)
% p(1:3) are translations (pixels); p(4:6) are scaling (%); p(7:12)
% are shears (degrees)
% voxsz = dimensions of voxels. Important for shears.

% Set up motion values depending on mocodims
p = fixp(p_in,mocodims);

% Determine shift of outermost pixel for shears.
shears = p(7:12)*pi/180;
p_sh_1wrt3 = (sz(3) / 2) * tan( shears(1) ) * voxsz(3)/voxsz(1);
p_sh_1wrt2 = (sz(2) / 2) * tan( shears(2) ) * voxsz(2)/voxsz(1);
p_sh_2wrt1 = (sz(1) / 2) * tan( shears(3) ) * voxsz(1)/voxsz(2);
p_sh_2wrt3 = (sz(3) / 2) * tan( shears(4) ) * voxsz(3)/voxsz(2);
p_sh_3wrt1 = (sz(1) / 2) * tan( shears(5) ) * voxsz(1)/voxsz(3);
p_sh_3wrt2 = (sz(2) / 2) * tan( shears(6) ) * voxsz(2)/voxsz(3);


% Apply affine transform in a "3-pass" manner (see Thevenaz and Unser,
% IEEE Int. Conf. Im Proc 1997). This is valid assuming that the
% transformation matrix is close to an identity matrix (which should be
% the case for the motion we expect to see).

% First pass scale
if isequal(p(4), p_a(4))
  img_curr = sub1;
elseif p(4)~=0
  img_curr = sinc_interp(img_curr_a,basic_scl_1*(1+p(4)/100),basic_scl_1,1);
  sub1 = img_curr;
else
  img_curr = img_curr_a;
  sub1 = img_curr;
end
% First pass shear
if isequal(p([4,7,8]), p_a([4,7,8]))
  img_curr = sub2;
elseif (p(7) ~= 0) || (p(8) ~= 0)
  img_curr = fft(img_curr,[],1);
  pmap = basic_1wrt3 * p_sh_1wrt3 + basic_1wrt2 * p_sh_1wrt2;
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],1);
  sub2 = img_curr;
else
  sub2 = img_curr;
end

% Second pass scale
if isequal(p([4,7,8,5]), p_a([4,7,8,5]))
  img_curr = sub3;
elseif p(5)~=0
  img_curr = sinc_interp(img_curr,basic_scl_2*(1+p(5)/100),basic_scl_2,2);
  sub3 = img_curr;
else
  sub3 = img_curr;
end
% Second pass shear
if isequal(p([4,7,8,5,9,10]), p_a([4,7,8,5,9,10]))
  img_curr = sub4;
elseif (p(9) ~= 0) || (p(10) ~= 0)
  img_curr = fft(img_curr,[],2);
  pmap = basic_2wrt1 * p_sh_2wrt1 + basic_2wrt3 * p_sh_2wrt3;
  img_curr = img_curr .* exp(1i*pmap);
  img_curr = ifft(img_curr,[],2);
  sub4 = img_curr;
else
  sub4 = img_curr;
end

% Third pass scale
if isequal(p([4,7,8,5,9,10,6]), p_a([4,7,8,5,9,10,6]))
  img_curr = sub5;
elseif (p(6)~=0)
  img_curr = sinc_interp(img_curr,basic_scl_3*(1+p(6)/100),basic_scl_3,3);
  sub5 = img_curr;
else
  sub5 = img_curr;
end
% Third pass shear
if (p(11) ~= 0) || (p(12) ~= 0)
  img_curr = fft(img_curr,[],3);
  pmap = basic_3wrt1 * p_sh_3wrt1 +  basic_3wrt2 * p_sh_3wrt2;
  img_curr = img_curr .* exp(1i*pmap);
  %img_curr = ifft(img_curr,[],3);
end


% Shifts
if any(p(1:3) ~= 0)
  if ~(p(11)~=0 || p(12)~=0)
    img_curr = fft(img_curr,[],3);
  end
  img_curr = fft(fft(img_curr,[],1),[],2);
  if isequal(p(1:3),p_a(1:3))
    pmap = pmap1;
  else
    pmap = basic_shf_1 * p(1) + basic_shf_2 * p(2) + basic_shf_3 * p(3);
    pmap = exp(1i*pmap);
    pmap1 = pmap;
  end
  img_curr = img_curr .* pmap;
  img_curr = ifft(ifft(ifft(img_curr,[],1),[],2),[],3);
elseif (p(11)~=0 || p(12)~=0)
  img_curr = ifft(img_curr,[],3);
end

% save motion params for next iteration to determine if we can reuse data.
p_a = p;

if doabs
    img_curr = abs(img_curr);
end

% Calculate cost function
Q = calccost(costfcn_type,img_curr,img_ref_a,immask);

end

%% nested subfunction: for 3D iteration using spatial domain interpolation instead of
% fourier transform based shifts and rotations. Useful for slice-acquired
% data with Gibbs ringing.
% Note: very slow.
function Q = rb_xform_spatial(p_in)
  % p(1:3) are translations (pixels); p(4:6) are rotations (degrees)
  % voxsz = dimensions of voxels. Important for rotation.
  
  % Set up motion values depending on mocodims
  p = fixp(p_in,mocodims);
  
  % Apply transformation
  img_curr = apply3DTrans(img_curr_a,p(1:6),voxsz,'makima',useGPU);
  
  % Set NaNs to 0
  img_curr(isnan(img_curr)) = 0;
  
  % Save current transformation
  p_a = p;
  
  % Calculate cost function
  Q = calccost(costfcn_type,img_curr,img_ref_a,immask);
  
end


end


function p_out = fixp(p_in,mocodims)

switch mocodims
    case '3Dshift'
        p_out = [p_in(1:3) 0 0 0 0 0 0 0 0 0];
    case '2D'
        p_out = [p_in(1:2) 0 0 0 p_in(3) 0 0 0 0 0 0];
    case '3D'
        p_out = [p_in(1:6) 0 0 0 0 0 0];
    otherwise
        p_out = p_in;
end

end


function [basic_2wrt3,basic_3wrt2,basic_1wrt3,basic_3wrt1,basic_1wrt2,basic_2wrt1,...
         basic_shf_1,basic_shf_2,basic_shf_3,basic_scl_1,basic_scl_2,basic_scl_3] =...
         createbasics(sz,cntr_a,classin)

if strcmp(classin,'gpuArray')
    sz = gpuArray(sz);     
    cntr_a = gpuArray(cntr_a); 
else
    sz = cast(sz,classin);     
    cntr_a = cast(cntr_a,classin); 
end

% Create basic scale indices
if strcmp(classin,'gpuArray')
    basic_scl_1 = gpuArray.colon(1,sz(1))-cntr_a(1)-1;
    basic_scl_2 = gpuArray.colon(1,sz(2))-cntr_a(2)-1;
    basic_scl_3 = gpuArray.colon(1,sz(3))-cntr_a(3)-1;
else
    basic_scl_1 = (1:sz(1))-cntr_a(1)-1;
    basic_scl_2 = (1:sz(2))-cntr_a(2)-1;
    basic_scl_3 = (1:sz(3))-cntr_a(3)-1;
    basic_scl_1 = cast(basic_scl_1,classin);
    basic_scl_2 = cast(basic_scl_2,classin);
    basic_scl_3 = cast(basic_scl_3,classin);
end

% Create basic shear matrices
if strcmp(classin,'gpuArray')
    mod_slp = (gpuArray.colon(1,sz(1)) - cntr_a(1)) / (sz(1)/2);
    basic_2wrt1 = mod_slp' * (gpuArray.colon(-sz(2)/2,(sz(2)/2-1)) / (sz(2)/2) * pi);
    basic_3wrt1 = mod_slp' * (gpuArray.colon(-sz(3)/2,(sz(3)/2-1)) / (sz(3)/2) * pi);
    basic_3wrt1 = reshape(basic_3wrt1, sz(1), 1, sz(3));
    %
    mod_slp = (gpuArray.colon(1,sz(2)) - cntr_a(2)) / (sz(2)/2);
    basic_1wrt2 = (gpuArray.colon(-sz(1)/2,(sz(1)/2-1)) / (sz(1)/2) * pi)' * mod_slp;
    basic_3wrt2 = mod_slp' * (gpuArray.colon(-sz(3)/2,(sz(3)/2-1)) / (sz(3)/2) * pi);
    basic_3wrt2 = reshape(basic_3wrt2, 1, sz(2), sz(3));
    %
    mod_slp = (gpuArray.colon(1,sz(3)) - cntr_a(3)) / (sz(3)/2);
    basic_1wrt3 = (gpuArray.colon(-sz(1)/2,(sz(1)/2-1)) / (sz(1)/2) * pi)' * mod_slp;
    basic_2wrt3 = (gpuArray.colon(-sz(2)/2,(sz(2)/2-1)) / (sz(2)/2) * pi)' * mod_slp;
    basic_1wrt3 = reshape(basic_1wrt3, sz(1), 1, sz(3));
    basic_2wrt3 = reshape(basic_2wrt3, 1, sz(2), sz(3));
else
    mod_slp = (colon(1,sz(1)) - cntr_a(1)) / (sz(1)/2);
    basic_2wrt1 = mod_slp' * (colon(-sz(2)/2,(sz(2)/2-1)) / (sz(2)/2) * pi);
    basic_3wrt1 = mod_slp' * (colon(-sz(3)/2,(sz(3)/2-1)) / (sz(3)/2) * pi);
    basic_3wrt1 = reshape(basic_3wrt1, sz(1), 1, sz(3));
    %
    mod_slp = (colon(1,sz(2)) - cntr_a(2)) / (sz(2)/2);
    basic_1wrt2 = (colon(-sz(1)/2,(sz(1)/2-1)) / (sz(1)/2) * pi)' * mod_slp;
    basic_3wrt2 = mod_slp' * (colon(-sz(3)/2,(sz(3)/2-1)) / (sz(3)/2) * pi);
    basic_3wrt2 = reshape(basic_3wrt2, 1, sz(2), sz(3));
    %
    mod_slp = (colon(1,sz(3)) - cntr_a(3)) / (sz(3)/2);
    basic_1wrt3 = (colon(-sz(1)/2,(sz(1)/2-1)) / (sz(1)/2) * pi)' * mod_slp;
    basic_2wrt3 = (colon(-sz(2)/2,(sz(2)/2-1)) / (sz(2)/2) * pi)' * mod_slp;
    basic_1wrt3 = reshape(basic_1wrt3, sz(1), 1, sz(3));
    basic_2wrt3 = reshape(basic_2wrt3, 1, sz(2), sz(3));    
end 
% if (0)
% basic_2wrt3_a = zeros(1,sz(2),sz(3),classin);
% basic_3wrt2_a = zeros(1,sz(2),sz(3),classin);
% basic_1wrt3_a = zeros(sz(1),1,sz(3),classin);
% basic_3wrt1_a = zeros(sz(1),1,sz(3),classin);
% basic_1wrt2_a = zeros(sz(1),sz(2),1,classin);
% basic_2wrt1_a = zeros(sz(1),sz(2),1,classin);
% for n = 1:sz(1)
%     mod_slp = (n - cntr_a(1)) / (sz(1)/2);
%     basic_2wrt1_a(n,:,1) = (-sz(2)/2:(sz(2)/2-1)) / (sz(2)/2) * pi * mod_slp;
%     basic_3wrt1_a(n,1,:) = (-sz(3)/2:(sz(3)/2-1)) / (sz(3)/2) * pi * mod_slp;
% end
% for n = 1:sz(2)
%     mod_slp = (n - cntr_a(2)) / (sz(2)/2);
%     basic_1wrt2_a(:,n,1) = (-sz(1)/2:(sz(1)/2-1)) / (sz(1)/2) * pi * mod_slp;
%     basic_3wrt2_a(1,n,:) = (-sz(3)/2:(sz(3)/2-1)) / (sz(3)/2) * pi * mod_slp;
% end
% for n = 1:sz(3)
%     mod_slp = (n - cntr_a(3)) / (sz(3)/2);
%     basic_1wrt3_a(:,1,n) = (-sz(1)/2:(sz(1)/2-1)) / (sz(1)/2) * pi * mod_slp;
%     basic_2wrt3_a(1,:,n) = (-sz(2)/2:(sz(2)/2-1)) / (sz(2)/2) * pi * mod_slp;
% end
% end
% basic_2wrt3 = repmat(basic_2wrt3, [sz(1) 1 1]);
% basic_3wrt2 = repmat(basic_3wrt2, [sz(1) 1 1]);
% basic_1wrt3 = repmat(basic_1wrt3, [1 sz(2) 1]);
% basic_3wrt1 = repmat(basic_3wrt1, [1 sz(2) 1]);
% basic_1wrt2 = repmat(basic_1wrt2, [1 1 sz(3)]);
% basic_2wrt1 = repmat(basic_2wrt1, [1 1 sz(3)]);

% Create basic shift matrices
% basic_shf_1 = zeros(sz(1),1,1,classin);
% basic_shf_2 = zeros(1,sz(2),1,classin);
% basic_shf_3 = zeros(1,1,sz(3),classin);
basic_shf_1(:,1,1) = (-sz(1)/2:(sz(1)/2-1)) / sz(1) * 2 * pi;
basic_shf_2(1,:,1) = (-sz(2)/2:(sz(2)/2-1)) / sz(2) * 2 * pi;
basic_shf_3(1,1,:) = (-sz(3)/2:(sz(3)/2-1)) / sz(3) * 2 * pi;
% basic_shf_1 = repmat(basic_shf_1, [1 sz(2) sz(3)]);
% basic_shf_2 = repmat(basic_shf_2, [sz(1) 1 sz(3)]);
% basic_shf_3 = repmat(basic_shf_3, [sz(1) sz(2) 1]);

% Apply fftshifts
basic_2wrt3 = fftshift(basic_2wrt3,2);
basic_3wrt2 = fftshift(basic_3wrt2,3);
basic_1wrt3 = fftshift(basic_1wrt3,1);
basic_3wrt1 = fftshift(basic_3wrt1,3);
basic_1wrt2 = fftshift(basic_1wrt2,1);
basic_2wrt1 = fftshift(basic_2wrt1,2);
basic_shf_1 = fftshift(basic_shf_1,1);
basic_shf_2 = fftshift(basic_shf_2,2);
basic_shf_3 = fftshift(basic_shf_3,3);

end

function Q = calccost(costfcn_type,img_curr,img_ref_a,immask)
% Calculate cost function

if costfcn_type == 1
    tmp = abs(img_curr(:).*img_ref_a(:));
    if ~isempty(immask)
        tmp = tmp.*immask(:);
    end
    Q = 1/mean(tmp);
elseif costfcn_type == 2
    if ~isempty(immask)
        Q = 1/information_normalized(real(img_curr(immask)).',real(img_ref_a(immask)).');
        if ~isreal(Q)
          Q = Q + 1/information_normalized(imag(img_curr(immask)).',imag(img_ref_a(immask)).');
        end
    else
        Q = 1/information_normalized(real(img_curr(:)).',real(img_ref_a(:)).');
        if ~isreal(Q)
          Q = Q + 1/information_normalized(imag(img_curr(:)).',imag(img_ref_a(:)).');
        end
    end
elseif costfcn_type == 3
    tmp = img_curr(:) - img_ref_a(:);
    tmp = tmp.*conj(tmp);
    if ~isempty(immask)
        tmp = tmp.*immask(:);
    end
    Q = sqrt(mean(tmp));
end

if isa(Q,'gpuArray')
    Q = gather(Q);
    Q = abs(Q);
end

end
