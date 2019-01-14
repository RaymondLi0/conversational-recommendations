import csv
import random
import argparse
import os
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("movielens_path")
args = parser.parse_args()

movielens_path = args.movielens_path

with zipfile.ZipFile(os.path.join(movielens_path, "ml-latest.zip"), 'r') as z:
    z.extractall(path=movielens_path)

data = []
with open(os.path.join(movielens_path, "ml-latest/ratings.csv"), "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != "userId":
            data.append(row)

random.shuffle(data)
n_data = len(data)

with open(os.path.join(movielens_path, "train_data"), 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in data[:int(.8 * n_data)]:
        writer.writerow(dat)
with open(os.path.join(movielens_path, "valid_data"), 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in data[int(.8 * n_data):int(.9 * n_data)]:
        writer.writerow(dat)
with open(os.path.join(movielens_path, "test_data"), 'w') as outfile:
    writer = csv.writer(outfile)
    for dat in data[int(.9 * n_data):]:
        writer.writerow(dat)
