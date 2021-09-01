import glob
import numpy as np
import pdb
import sys
import os

assert len(sys.argv) == 3, "model dir | acc, f1, mcc, corr"

idx = 3 if sys.argv[2] in ["acc", "corr"] else 5

fns = glob.glob(os.path.join(sys.argv[1], '*_eval_results.txt'))
fns.sort()
print(fns)

scores = []
data = []
for fn in fns:
    s = open(fn).read().splitlines()[idx].split(' = ')[-1]
    scores.append(float(s))
    data.append([fn, round(float(s), 4)])

np.save(os.path.join(sys.argv[1], 'all_results.npy'), np.array(data)[:,1])

data.sort(key=lambda tup: tup[1], reverse=True)
for d in data:
    print(d)

print("\nMax: {:.3f}, Min {:.3f}, Mean {:.3f}, Std {:.2f}"
        .format(np.max(scores), np.min(scores), np.mean(scores), np.std(scores)))
