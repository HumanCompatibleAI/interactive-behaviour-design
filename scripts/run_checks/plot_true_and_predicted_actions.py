import argparse

from pylab import *

parser = argparse.ArgumentParser()
parser.add_argument('log')
args = parser.parse_args()

with open(args.log, 'r') as f:
    lines = f.read().rstrip().split('\n')

ion()
for episode_n in range(1, 100):
    clf()
    grid()
    title(f"Episode {episode_n}")

    elines = None
    for n, l in enumerate(lines):
        if f'Episode {episode_n}' in l:
            elines = lines[n+1:n+1+300]

    data = np.array([list(map(float, l.split(' '))) for l in elines])
    plot(data[:, 0], label='Environment reward')

    plot(data[:, 1], label='Predicted reward')

    env_rew_min, env_rew_max = min(data[:, 0]), max(data[:, 0])
    delta = env_rew_max - env_rew_min
    pred = data[:, 1]
    pred = (pred - min(pred)) / (max(pred) - min(pred)) * (env_rew_max - env_rew_min) + env_rew_min
    plot(pred, label='Predicted reward (rescaled)')
    legend()
    try:
        waitforbuttonpress()
    except KeyboardInterrupt:
        break