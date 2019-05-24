import argparse
import pickle

from lunarlander_manual import Demonstration
from utils import save_video

parser = argparse.ArgumentParser()
parser.add_argument('demonstrations_pkl')
args = parser.parse_args()

Demonstration
with open(args.demonstrations_pkl, 'rb') as f:
    demonstrations = pickle.load(f)

print(len(demonstrations))
save_video(demonstrations[0].frames, 'demo.mp4')
