from main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str, default='data/philosophy.txt')
parser.add_argument('--model', type=str)
args = parser.parse_args()

gm = GenModel(args.input_dir, args.model)
gm.fit()
