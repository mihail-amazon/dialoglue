import json
import numpy as np

np.random.seed(0)

train_dials = json.load(open("dialoglue/multiwoz/train_dials.json"))
train_dials_few_keys = list(np.random.choice(list(train_dials.keys()), int(len(train_dials)/10)))
train_dials_few = {k:train_dials[k] for k in train_dials_few_keys}
json.dump(train_dials_few, open("dialoglue/multiwoz/train_dials_10.json", "w+"))
