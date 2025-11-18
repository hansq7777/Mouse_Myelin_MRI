%run ogse2freq

path1 = '/srv/baron/trainees/nrahman/DataPreproc/ConcussionStudy/Shams/';
path2 = 'OGSE_5Shapes_1A_5Rep_TR10/OGSE_5Shapes_1A_5Rep_TR10_aveComb_preproc';
mousenum = ["NR13_F/","NR14_F/","NR15_F/","NR16_F/","NR17_F/","NR18_F/"];
time = ["8week/","20week/"];

freq = [0,50,100,145,190];
directions = 10;

for n = 1:6
    for t = 1:2

        niftiName = sprintf('%s%s%s%s',path1,mousenum(n),time(t),path2);
        maskName = sprintf('%s_mask_after',niftiName);
        bvalName = niftiName;

        %split into each freq
        ogse2freq(niftiName, maskName, bvalName, freq, directions);

    end
end
    