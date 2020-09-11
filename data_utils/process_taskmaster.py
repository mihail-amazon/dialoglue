import json
import numpy as np
import os
import sys

# set seed
np.random.seed(0)

dataset = sys.argv[1]
if not dataset.endswith("/"):
    dataset = dataset + "/"


train = []
val = []
test = []
for fn in os.listdir(dataset):
    cur_data = json.load(open(dataset + fn))
    np.random.shuffle(cur_data)    

    train += cur_data[:int(len(cur_data)*0.8)]
    val += cur_data[int(len(cur_data)*0.8):int(len(cur_data)*0.9)]
    test += cur_data[int(len(cur_data)*0.9):]

json.dump(train, open(dataset+"train.json", "w+"))
json.dump(val, open(dataset+"val.json", "w+"))
json.dump(test, open(dataset+"test.json", "w+"))

# Get full list of slots
slots = set()
for conv in train:
    for utt in conv['utterances']:
        for segment in utt.get('segments', []):
            for ann in segment['annotations']:
                slots.add(ann['name'])

all_slots = [pre+slot for slot in slots for pre in ["B-", "I-"]]
new_slots = []
for slot in all_slots:
    new_slots.append("".join(c for c in slot if not c.isdigit() and c != " "))

new_slots = ["O"] + list(set(new_slots))
json.dump(new_slots, open(dataset+"slots.json", "w+"))
