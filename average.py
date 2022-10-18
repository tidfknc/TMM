import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', type=str, help='result file path')

args = parser.parse_args()


res = []
with open(args.file) as f:
    for line in f:
        line = line.strip()
        res.append(float(line))

print("mean WA.F1 on five random runs: ", np.mean(res))
print("std WA.F1 on five random runs: ", np.std(res))
