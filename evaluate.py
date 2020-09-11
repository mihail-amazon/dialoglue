import json
import re
import sys

from collections import defaultdict, namedtuple

def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    """
    DialoGLUE evaluation function for eval.ai 

    See documentation: https://evalai.readthedocs.io/en/latest/evaluation_scripts.html
    See test_annotation_file: gt_test.json

    The same annotation file and function will be used regardless of mode (e.g., full data, few shot, few shot + unlabeled)
    """
    gt_outputs = json.load(open(test_annotation_file))    
    gen_outputs = json.load(open(user_annotation_file))    

    # Iterate over the tasks/datsets
    dataset_to_intent = {
        "hwu": "intent",
        "clinc": "intent",
        "banking": "intent",
        "restaurant8k": "slot",
        "dstc8_sgd": "slot",
        "taskmaster": "slot",
        "multiwoz": "dst",
        "top": "top",
    }
    results = {}
    for dataset, gt_outputs in gt_outputs.items():
        # Calculate score differently depending on the dataset/task
        task = dataset_to_intent[dataset]

        if task == "intent":
            # Calculate accuracy between generated and ground-truth
            results[dataset] = sum(p == t for p,t in zip(gen_outputs.get(dataset), gt_outputs))/len(pred)
        elif task == "slot":
            # Use Python equivalent of the conll evaluation script
            words, gt_slots = gt_outputs
            results[dataset] = conlleval(words, slots, gen_outputs.get(dataset))
        elif task == "dst":
            # Joint accuracy for DST.
            joint_acc = 0
            for p,t in zip(gen_outputs.get(dataset), gt_outputs):
                if p == t:
                    joint_acc += 1

            results[datset] = joint_acc/len(gt_outputs)
        elif task == "top":
            # Exact match accuracy for TOP
            results[dataset] =  sum(p == t for p,t in zip(gen_outputs.get(dataset), gt_outputs))/len(pred)

    return {"result": {"test_split": results}}



# Python version of the evaluation script from CoNLL'00-
# Retrieved from: https://raw.githubusercontent.com/spyysalo/conlleval.py/master/conlleval.py
class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')

        features = line.split()

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct.keys()) + list(c.t_found_guessed.keys())):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type

def report(counts, out=None):
    overall, _ = metrics(counts)
    return overall.fscore

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def conlleval(words, gt_slots, pred_slots):
    # Create a list of inputs for the conll eval script
    input_list = []

    # Iterate over the sentences
    for i in range(len(words)):
        # Add a BOS
        input_list.append("BOS O O")

        # Iterate over the words
        for word, gt_slot, pred_slot in zip(words[i], gt_slots[i], pred_slots[i]):
            input_list.append(word + " " + gt_slot + " " + pred_slot)

        # Add an EOS
        input_list.append("EOS O O")

        # Add an empty line
        input_list.append("")


    counts = evaluate(input_list)
    return report(counts)
