import random

random.seed(0)
# select 300 lines from the file

test_file_original = "dataset/test.hard.json"
test_file_sample = "dataset/test.hard.sample300.json"
with open(test_file_original, "r") as f:
    lines = f.readlines()
random.shuffle(lines)
lines = lines[:300]
with open(test_file_sample, "w") as f:
    f.writelines(lines)