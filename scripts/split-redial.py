import zipfile
import json
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("redial_path")
args = parser.parse_args()

redial_path = args.redial_path

with zipfile.ZipFile(os.path.join(redial_path, "redial_dataset.zip"), 'r') as z:
    z.extractall(path=redial_path)

data = []
for line in open(os.path.join(redial_path, "train_data.jsonl")):
    data.append(json.loads(line))
random.shuffle(data)
n_data = len(data)
split_data = [data[:int(.8 * n_data)], data[int(.8 * n_data):]]

with open(os.path.join(redial_path, "train_data"), 'w') as outfile:
    for example in split_data[0]:
        json.dump(example, outfile)
        outfile.write('\n')
with open(os.path.join(redial_path, "valid_data"), 'w') as outfile:
    for example in split_data[1]:
        json.dump(example, outfile)
        outfile.write('\n')
