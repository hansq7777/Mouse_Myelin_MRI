function MTsat = calcMTsat(refPD,refT1,MTon,alphPD,alphT1,TRPD,TRT1,RFlocal,dofilt)
%Function to calculate MTsat - called by nii2mtsat.m
%refPD: reference PDw scan
%refT1: reference T1w scan
%MTon: MTw scan
%alphPD: flip angle in reference PDw scan
%alphT1: flip angle in reference T1w scan
%TRPD: TR (repetition time) in PDw scan
%TRT1: TR in T1w scan
%RFlocal: relative local flip angle compared to nominal flip angle
%dofilt: NaN filtering (0 or 1); default is 1

    if nargin<9
        dofilt = 1;
    end

    %Equarions from Helms et al. 2008 (MRM)
    R1app = 0.5*(refT1*alphT1/TRT1-refPD*alphPD/TRPD)./(refPD/alphPD-refT1/alphT1);
    Aapp = (TRPD*alphT1/alphPD-TRT1*alphPD/alphT1)*refPD.*refT1./...
        (refT1*TRPD*alphT1 - refPD*TRT1*alphPD);

    MTsatApp = (Aapp*alphPD./MTon-1).*R1app*TRPD-alphPD^2/2;
    
    if ~isempty(RFlocal)
        %Apply B1 inhomogeneity correction according to Hagiwara et al. 2018
        MTsat = (MTsatApp*(1-0.4))./(1-0.4*RFlocal);
    else
        MTsat = MTsatApp;
    end

    % Filter nan values
    if dofilt
        nanmask = or(isnan(MTsat), isinf(MTsat));
        filt = ones([3 3 3]);
        filt(2,2,2) = 0;
        filt = filt/sum(filt(:));
        MTsat(nanmask) = 0;
        MTsat_filt = convn(MTsat,filt,'same');
        MTsat(nanmask) = MTsat_filt(nanmask);
    end

end