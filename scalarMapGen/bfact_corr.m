function bfact_corr(bvalName,bfactName)
%Function to apply corrections to b-values of OGSE frequencies, based on
%b-values acquired from OGSE scan on a water phantom
%bvalName: name of file with b-values (exclude extension .bval)
%bfactName: name of file with b-value calibration values from water phantom

bfact = dlmread(bfactName);

bval = dlmread(sprintf('%s.bval', bvalName));
bval = bval(:);

shapes = 5;
directions = 11;

bval_new = bval;

for sh = 1:shapes
    counter = sh;               %counter will go to the correct image volume for a certain direction
    for d = 1:directions
        bval_new(counter) = bval(counter)*bfact(d)      %apply calibration correction
        counter = counter + shapes;
    end
end

%save new bval and old bval
[st, sysOut] = system(sprintf('cp %s.bval %s_old.bval', bvalName, bvalName)); if st; error(sysOut); end
bval_new = bval_new';
dlmwrite(sprintf('%s.bval',bvalName), bval_new, 'delimiter',' ');

end
