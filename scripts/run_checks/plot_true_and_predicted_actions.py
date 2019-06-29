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
    plot(data[:, 0], label='Environment reward')

    data[:, 1] += np.random.randint(-5, 5)
    data[:, 1] *= np.random.rand() * 2

    plot(data[:, 1], label='Predicted reward')

    env_rew_min, env_rew_max = min(data[:, 0]), max(data[:, 0])
    delta = env_rew_max - env_rew_min
    predicted_rewards = data[:, 1]
    predicted_rewards = (predicted_rewards - min(predicted_rewards)) / (max(predicted_rewards) - min(predicted_rewards)) * (env_rew_max - env_rew_min) + env_rew_min
    plot(predicted_rewards, label='Predicted reward (rescaled)')

    predicted_rewards = data[:, 1]
    predicted_rewards = log(exp(predicted_rewards - logsumexp(predicted_rewards)))
    plot(predicted_rewards, label='Predicted reward (Boltzmann)')

    ylim([-10, 4])


    legend()
    s = input()
    if s == 'q':
        break
