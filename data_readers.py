import csv
import logging
import json
import numpy as np
import os
import pickle
import spacy

from collections import defaultdict
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict

from constants import SPECIAL_TOKENS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class ResponseSelectionDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_response_selection_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            for example in tqdm(data, desc="Preprocessing"):
                context = ' <turn> '.join([e['utterance'] for e in example['messages-so-far']])
                context = ' '.join(context.split()[-max_seq_length:])
                encoded_context = tokenizer.encode(context)

                response = example['options-for-correct-answers'][0]['utterance']
                response = ' '.join(response.split()[-max_seq_length:])
                encoded_response = tokenizer.encode(response)

                candidates = [
                    ' '.join(e['utterance'].split()[-max_seq_length:])
                    for e in example['options-for-next']
                ]
                encoded_candidates = [tokenizer.encode(cand) for cand in candidates]

                correct_id = example['options-for-correct-answers'][0]['candidate-id']
                correct_ind = [
                    i for i,e in enumerate(example['options-for-next'])
                    if e['candidate-id'] == correct_id
                ]

                candidate_inputs = [{
                    "input_ids": np.array(cand.ids),
                    "attention_mask": np.array(cand.attention_mask),
                    "token_type_ids": np.array(cand.type_ids)
                } for cand in encoded_candidates]

                self.examples.append({
                    "ctx_input_ids": np.array(encoded_context.ids),
                    "ctx_attention_mask": np.array(encoded_context.attention_mask),
                    "ctx_token_type_ids": np.array(encoded_context.type_ids),
                    "rsp_input_ids": np.array(encoded_response.ids),
                    "rsp_attention_mask": np.array(encoded_response.attention_mask),
                    "rsp_token_type_ids": np.array(encoded_response.type_ids),
                    "candidates": candidate_inputs,
                    "correct_candidate": correct_ind
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class StateTrackingDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))


        # Slot value categories
        slot_vocab_path = os.path.join(data_dirname, "slot_vocabs.json")
        slot_values = json.load(open(slot_vocab_path))

        # Add -empty as a value for each slot
        slot_values = {k: v + ['-empty'] for k,v in slot_values.items()}
        
        self.slot_label_to_idx = { key: dict((label, idx) for idx, label in enumerate(labels)) for key,labels in slot_values.items() }
        self.idx_to_slot_label = slot_values

        # Slot key categories
        labels = slot_values.keys()
        self.label_to_idx = dict((label, idx) for idx, label in enumerate(labels))
        self.idx_to_label = sorted(self.label_to_idx, key=self.label_to_idx.get)

        # Length of vocab for each slot
        self.slot_lengths = [ len(self.slot_label_to_idx[slot]) for slot in self.idx_to_label ]

        def _slot_value(belief_state, slot):
            for text in belief_state:
                if text.startswith(slot):
                  return text.replace(slot, "")

            return "-empty"

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_dst_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            last_belief = []
            last_id = ""
            for example in tqdm(data):
                # Encode the dialog history
                history = ' '.join(example['dialog_history'].replace(';', '<turn>').split())

                last_utt = history.split('<turn>')[-2].strip()
                last_sys_utt = history.split('<turn>')[-3].strip()
                prev_dialog = '<turn>'.join(history.split('<turn>')[1:-3]).strip()

                history = prev_dialog + ' <turn> ' + last_sys_utt + ' <turn> ' + last_utt

                encoded = tokenizer.encode(history)

                belief_state = example['turn_belief']
                if last_id == example['ID']:
                    belief_state = [e for e in example['turn_belief'] if e not in last_belief]

                last_id = example['ID']
                last_belief = example['turn_belief']

                full_state_labels = []
                state_labels = []
                for slot in self.label_to_idx:
                  value = _slot_value(example['turn_belief'], slot)
                  full_state_labels.append(self.slot_label_to_idx[slot].get(value, self.slot_label_to_idx[slot]['-empty']))

                  value = _slot_value(belief_state, slot)
                  state_labels.append(self.slot_label_to_idx[slot].get(value, self.slot_label_to_idx[slot]['-empty']))

                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "state_label": np.array(state_labels),
                    "original_state_label": example['turn_belief'],
                    "ID": last_id
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class IntentDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_intent_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            # Load spacy
            nlp = spacy.load("en_core_web_sm")

            self.examples = []
            reader = csv.reader(open(data_path))
            next(reader, None)
            out = []
            for utt, intent in tqdm(reader):
                encoded = tokenizer.encode(utt)

                if sum([e==0 for e in encoded.ids]) < 20:
                    out.append(sum([e!=0 for e in encoded.ids])+20)
                    print("Raise max length to for observers: ", max(out))


                # Merge spacy tokenization into attention mask given tokenized input
                #words = nlp(utt)
                #spacy_word_lens = [len(w) for w in words]
                #tokenized_word_lens = [len(tokenizer.decode([w]).replace("##", "")) for w in encoded.ids if w > 0] 

                ## For each tokenizer token, determine the list of spacy tokens it maps to.
                #if sum(spacy_word_lens) != sum(tokenized_word_lens):
                #    import pdb; pdb.set_trace()
                #assert sum(spacy_word_lens) == sum(tokenized_word_lens), "Tokenization error"
                #
                ## Make ranges for each token
                #acc = 0
                #spacy_word_ranges = [0]
                #for length in spacy_word_lens:
                #    acc += length
                #    spacy_word_ranges.append(acc)

                #acc = 0
                #tokenized_word_ranges = [0]
                #for length in tokenized_word_lens:
                #    acc += length
                #    tokenized_word_ranges.append(acc)

                ## Can be optimized. But doing it this way in the interest of bug-free
                ## For each tokenized token, find all of the spacy tokens
                #token_map = defaultdict(list)
                #for t_i, t_start in enumerate(tokenized_word_ranges[:-1]):
                #    t_end = tokenized_word_ranges[t_i+1]
                #    t_inds = set(range(t_start, t_end))
                #    for s_i, s_start in enumerate(spacy_word_ranges[:-1]):
                #        s_end = spacy_word_ranges[s_i+1]
                #        s_inds = set(range(s_start, s_end))
                #        if s_inds.intersection(t_inds):
                #            token_map[t_i].append(s_i)
                #
                #noun_attention_map = [w.tag_[0] == 'N' for w in words]
                #tokenized_noun_attention_map = [any([noun_attention_map[j] for j in token_map[i]]) for i in range(len(encoded))]

                #verb_attention_map = [w.tag_[0] == 'V' for w in words]
                #tokenized_verb_attention_map = [any([verb_attention_map[j] for j in token_map[i]]) for i in range(len(encoded))]

                #ids = [e if e > 0 else 103 for e in encoded.ids ]
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    #"input_ids": np.array(ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    #"attention_mask": np.ones(len(encoded.attention_mask))[-max_seq_length:],
                    #"noun_attention_mask": np.array(tokenized_noun_attention_map).astype(int)[-max_seq_length:],
                    #"verb_attention_mask": np.array(tokenized_verb_attention_map).astype(int)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class SlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.txt")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots_cached".format(split, vocab_file_name))
        texts = []
        slotss = []
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            for example in tqdm(data):
                text, slots = self.parse_example(example) 
                texts.append(text)
                slotss.append(slots)
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "slot_labels": encoded_slot_labels[-max_seq_length:],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example):
        text = example['userInput']['text']

        # Create slots dictionary
        word_to_slot = {}
        for label in example.get('labels', []):
            slot = label['slot']
            start = label['valueSpan'].get('startIndex', 0)
            end = label['valueSpan'].get('endIndex', -1)

            for word in text[start:end].split():
                word_to_slot[word] = slot
          
        # Add context if it's there
        if 'context' in example:
            for req in example['context'].get('requestedSlots', []):
                text = req + " " + text

        # Create slots list
        slots = []
        cur = None
        for word in text.split():
            if word in word_to_slot:
                slot = word_to_slot[word]
                if cur is not None and slot == cur:
                    slots.append("I-" + slot) 
                else:
                    slots.append("B-" + slot) 
                    cur = slot
            else:
                slots.append("O")
                cur = None

        return text, " ".join(new_slots) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TMSlotDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "slots.json")
        slot_names = json.load(open(slot_vocab_path))
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_slots_cached".format(split, vocab_file_name))
        texts = []
        slotss = []
        if not os.path.exists(cached_path):
            self.examples = []
            data = json.load(open(data_path))
            for example in tqdm(data):
                for text,slots in  self.parse_example(example, max_seq_length):
                    encoded = tokenizer.encode(text)
                    encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                                   len(encoded.ids),
                                                                   tokenizer,
                                                                   self.slot_label_to_idx,
                                                                   max_seq_length)
                    self.examples.append({
                        "input_ids": np.array(encoded.ids)[-max_seq_length:],
                        "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                        "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                        "slot_labels": encoded_slot_labels[-max_seq_length:]
                    })
                    texts.append(text)
                    slotss.append(slots)
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))

            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def parse_example(self, example, max_seq_length):
        examples = []
        history = []
        for utt in example['utterances']:
            history.append(utt['speaker'].lower() + ' ' + utt['text'])
            if utt['speaker'] == 'ASSISTANT' or 'segments' not in utt:
                continue

            text = " <turn> ".join(history) 
            text = "".join([e if ord(e) < 128 else ' ' for e in text]) # remove non-ascii, since some unicode cause issues with tokenizer
            text = " ".join(text.split())
            text = " ".join(text.split()[-max_seq_length:])

            word_to_slot = {}
            for segment in utt['segments']:
                # If there are multiple slot keys, take the most popular one
                slot_possibilities = [ann['name'].replace(' ', '') for ann in segment['annotations']]
                slot_possibilities = ["".join(c for c in slot if not c.isdigit()) for  slot in slot_possibilities]
                slot = min(slot_possibilities, key=lambda k: self.slot_label_to_idx.get("B-" + k))

                for word in segment['text'].split():
                    word_to_slot[word] = slot
          
            # Create slots list
            slots = []
            cur = None
            for word in utt['text'].split():
                if word in word_to_slot:
                    slot = word_to_slot[word]
                    if cur is not None and slot == cur:
                        slots.append("I-" + slot) 
                    else:
                        slots.append("B-" + slot) 
                        cur = slot
                else:
                    slots.append("O")
                    cur = None

            # Pre-pad with Os for the other utterances
            slots = ["O"] * (len(text.split()) - len(slots)) + slots

            # Filter slots to remove numeric counts
            new_slots = []
            for slot in slots:
                new_slots.append("".join(c for c in slot if not c.isdigit()))

            # Add to list
            examples.append((text, " ".join(new_slots)))
    
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TOPDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 tokenizer: BertWordPieceTokenizer,
                 max_seq_length: int,
                 vocab_file_name: str):
        # For caching
        data_dirname = os.path.dirname(os.path.abspath(data_path))
        split = os.path.basename(os.path.abspath(data_path))

        # Slot categories
        slot_vocab_path = os.path.join(os.path.dirname(data_path), "vocab.slot")
        slot_names = [e.strip() for e in open(slot_vocab_path).readlines()]
        slot_names.insert(0, "[PAD]")
        self.slot_label_to_idx = dict((label, idx) for idx, label in enumerate(slot_names))
        self.slot_idx_to_label = {idx: label for label, idx in self.slot_label_to_idx.items()}

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        self.intent_label_to_idx = dict((label, idx) for idx, label in enumerate(intent_names))
        self.intent_idx_to_label = {idx: label for label, idx in self.intent_label_to_idx.items()}

        # Process data
        self.tokenizer = tokenizer
        cached_path = os.path.join(data_dirname, "{}_{}_top_cached".format(split, vocab_file_name))
        if not os.path.exists(cached_path):
            self.examples = []
            data = [e.strip() for e in open(data_path).readlines() ]
            for example in tqdm(data):
                example, intent = example.split(" <=> ")
                text = " ".join([e.split(":")[0] for e in example.split()])
                slots = " ".join([e.split(":")[1] for e in example.split()])
                encoded = tokenizer.encode(text)
                encoded_slot_labels = self.encode_token_labels([text], [slots],
                                                               len(encoded.ids),
                                                               tokenizer,
                                                               self.slot_label_to_idx,
                                                               max_seq_length)
                self.examples.append({
                    "input_ids": np.array(encoded.ids)[-max_seq_length:],
                    "attention_mask": np.array(encoded.attention_mask)[-max_seq_length:],
                    "token_type_ids": np.array(encoded.type_ids)[-max_seq_length:],
                    "slot_labels": encoded_slot_labels[-max_seq_length:],
                    "intent_label": self.intent_label_to_idx[intent],
                    "ind": len(self.examples),
                })
            with open(cached_path, "wb") as f:
                pickle.dump(self.examples, f)
        else:
            LOGGER.info("Loading from cached path: {}".format(cached_path))
            with open(cached_path, "rb") as f:
                self.examples = pickle.load(f)

    def encode_token_labels(self,
                            text_sequences,
                            slot_names,
                            encoded_length,
                            tokenizer,
                            slot_map,
                            max_length) -> np.array:
    
        def _get_word_tokens_len(word: str, tokenizer: BertWordPieceTokenizer) -> int:
            return sum(map(lambda token: 1 if token not in SPECIAL_TOKENS else 0,
                           tokenizer.encode(word).tokens))
    
        encoded = np.zeros(shape=(len(text_sequences), encoded_length), dtype=np.int32)
        for i, (text_sequence, word_labels) in enumerate(
                zip(text_sequences, slot_names)):
            encoded_labels = []
            for word, word_label in zip(text_sequence.split(), word_labels.split()):
                encoded_labels.append(slot_map[word_label])
                expand_label = word_label.replace("B-", "I-")
                if not expand_label in slot_map:
                    expand_label = word_label
                word_tokens_len = _get_word_tokens_len(word, tokenizer)
                encoded_labels.extend([slot_map[expand_label]] * (word_tokens_len - 1))
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        return encoded.squeeze()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
