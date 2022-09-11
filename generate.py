from main import *
import argparse
import numpy as np

gm = GenModel('/content/drive/MyDrive/philosophy.txt', 'gg.pth')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--prefix', type=str, default=np.random.choice(gm.clean_text.split(), 1))
parser.add_argument('--length', type=int)
args = parser.parse_args()

pre = gm.generate(args.model, args.prefix, args.length)
print(pre)