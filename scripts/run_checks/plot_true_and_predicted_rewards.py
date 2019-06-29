import argparse
from time import sleep

from pylab import *
from scipy.special import logsumexp

parser = argparse.ArgumentParser()
parser.add_argument('log')
parser.add_argument('--start', type=int, default=1)
args = parser.parse_args()

with open(args.log, 'r') as f:
    lines = f.read().rstrip().split('\n')

ion()
for episode_n in range(args.start, 100000000000):
    clf()
    grid()
    title(f"Episode {episode_n}")

    elines = None
    for n, l in enumerate(lines):
        if f'Episode {episode_n}' in l or f'Test {episode_n}' in l:
            elines = lines[n+1:n+1+100]

    data = np.array([list(map(float, l.split(' '))) for l in elines])

    env_rewards = data[:, 0]
    pred_rewards = data[:, 1]
    pred_rewards_rescaled = np.copy(pred_rewards)

    scale = (max(env_rewards) - min(env_rewards)) / (max(pred_rewards) - min(pred_rewards))
    pred_rewards_rescaled *= scale
    shift = min(env_rewards) - min(pred_rewards_rescaled)
    pred_rewards_rescaled += shift

    pred_rewards_rescaled = pred_rewards * scale + shift
    print(scale, shift)

    print(np.mean(pred_rewards))
    print(np.std(pred_rewards))
    pred_rewards -= np.mean(pred_rewards)
    pred_rewards /= np.std(pred_rewards)

    plot(env_rewards, label='Environment reward')
    plot(pred_rewards, label='Predicted reward')
    plot(pred_rewards_rescaled, label='Predicted reward (rescaled)')

    # ylim([-10, 4])


    legend()
    s = input()
    if s == 'q':
        break
