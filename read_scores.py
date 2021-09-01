import numpy as np
import os
import sys
from glob import glob

assert len(sys.argv) == 2, "npy dir"
npy_dir = sys.argv[1]

fns = glob(os.path.join(npy_dir, "*.npy"))
fns.sort()

for fn in fns:
	print(fn.split('/')[-1])
	x = np.load(fn)
	x = np.array([float(i) for i in x])*100
	print(x)
	print("Mean {:.2f}, std {:.2f}, Max {:.2f}, Min {:.2f}".
		format(x.mean(), x.std(), x.max(), x.min()))
	print("----------------------------------------------------")

