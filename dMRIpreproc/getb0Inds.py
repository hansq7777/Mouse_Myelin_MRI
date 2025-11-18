# Script to find b0 indices

import sys

with open(sys.argv[1]) as f:
    fdata=f.read()
bdata = fdata.split()					

bvals = [float(s) for s in bdata]

bvecs = []
with open(sys.argv[2]) as f:
    for line in f.readlines():
        bv = line.split()
        bvecs.append([float(x) for x in bv])

binds = []
minb = min(bvals)
maxb = max(bvals)
if abs(maxb-minb) < 30:
    # Only one b-shell, so pick all averages in a random direction
    for idx, b in enumerate(bvals):
        if bvecs[0][idx] == bvecs[0][0]:
            if bvecs[1][idx] == bvecs[1][0]:
                if bvecs[2][idx] == bvecs[2][0]:
                    binds.append(idx)
else:
	for idx, b in enumerate(bvals):
	    if abs(b-minb) < 30:
                binds.append(idx)

for i in binds:
    print(i)
