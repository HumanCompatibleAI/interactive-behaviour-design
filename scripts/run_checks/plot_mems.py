#!/usr/bin/env python3

import argparse
import subprocess
import tempfile
from collections import namedtuple

from pylab import *

plt.rcParams.update({'figure.max_open_warning': 0})

parser = argparse.ArgumentParser()
parser.add_argument('mem_logs', nargs='*')
args = parser.parse_args()

Log = namedtuple('Log', 'name values timestamps')

with tempfile.TemporaryDirectory() as d:
    mem_logs = []
    for log_file in args.mem_logs:
        log_name = os.path.basename(log_file)
        with open(log_file, 'r') as f:
            lines = f.read().rstrip().split('\n')
        values = [float(l.split()[1]) for l in lines]
        timestamps = [float(l.split()[2]) for l in lines]
        relative_timestamps = [t - timestamps[0] for t in timestamps]
        mem_logs.append(Log(log_name, values, timestamps))
    mem_logs.sort(key=lambda log: max(log.values))
    mem_logs = mem_logs[::-1]

    for i, mem_log in enumerate(mem_logs):
        print(mem_log.name)
        figure()
        title(mem_log.name)
        plot(mem_log.timestamps, mem_log.values)
        savefig(os.path.join(d, '{:03d}.png'.format(i)))
    subprocess.call(f'montage {d}/*.png -tile 6x -mode concatenate memory_plots.png', shell=True)
