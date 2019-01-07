# -*- coding: iso-8859-15 -*-

import collections
import random

import modeling, tokenization

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

    
# functions to parse data
def create_masked_lm_predictions_based_given(tokens, max_predictions_per_seq, segment_ids):
  """Creates the predictions for the masked LM objective."""

  tokens_len = len(tokens)

  output_tokens = []
  masked_lm_positions = []
  masked_lm_labels = []
  segment_ids_new = []
  i=0
  idx=0
  num_masks = 0
  while i < tokens_len:
    tok = tokens[i]
    if tok==u'01':
      masked_token = "[MASK]"
      output_tokens.append(masked_token)
      masked_lm_positions.append(idx)
      i+=1
      num_masks += 1
      masked_lm_labels.append(tokens[i])
      segment_ids_new.append(segment_ids[i])
      idx+=1
    else:
      output_tokens.append(tok)
      segment_ids_new.append(segment_ids[i])
      idx+=1
    i+=1
  if num_masks>max_predictions_per_seq:
    print ('too many masks')
  # print (tokens)
  # print (output_tokens)
  # print (masked_lm_positions)
  # print (masked_lm_labels)
  # abc

  return (output_tokens, masked_lm_positions, masked_lm_labels, segment_ids_new)


def generate_example_given_instance(instance, tokenizer, max_seq_length,
                                    max_predictions_per_seq):
    
  input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
  input_mask = [1] * len(input_ids)
  segment_ids = list(instance.segment_ids)
  assert len(input_ids) <= max_seq_length

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length, "%d != %d"%(len(segment_ids),max_seq_length) 

  masked_lm_positions = list(instance.masked_lm_positions)
  masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
  masked_lm_weights = [1.0] * len(masked_lm_ids)

  while len(masked_lm_positions) < max_predictions_per_seq:
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  next_sentence_label = 1 if instance.is_random_next else 0

  features = collections.OrderedDict()
  features["input_ids"] = input_ids
  features["input_mask"] = input_mask
  features["segment_ids"] = segment_ids
  features["masked_lm_positions"] = masked_lm_positions
  features["masked_lm_ids"] = masked_lm_ids
  features["masked_lm_weights"] = masked_lm_weights
  features["next_sentence_labels"] = [next_sentence_label]
  features['masked_lm_labels'] = np.ones_like(input_ids)*-1
  features['masked_lm_labels'][masked_lm_positions] = masked_lm_ids
  features['masked_lm_labels'][0] = -1

  return features



SENT_A = "Nancy has just got a 01 job as a secretary in a company. Monday was the first day she \
went to work, so she was very excited and arrived 01 early. "
SENT_B =  "She 01 pushed the door open and found nobody there. \"I am the 01 first to arrive.\" She \
thought and come to her desk. She was surprised to find a bunch of 01 flowers on it. "
IS_RANDOM_NEXT = True

VOCAB_FILE = '../../uncased_L-12_H-768_A-12/vocab.txt'
DO_LOWER_CASE = True
MAX_PREDICTIONS_PER_SEQ = 20
MAX_SEQ_LENGTH = 128
BERT_CONFIG_FILE = '../../uncased_L-12_H-768_A-12/bert_config.json'
INIT_DIRECTORY = "../../uncased_L-12_H-768_A-12"
INIT_CHECKPOINT_PT = "../../uncased_L-12_H-768_A-12/pytorch_model.bin"
LEARNING_RATE = 2e-5
NUM_TRAIN_STEPS = 1
NUM_WARMUP_STEPS = 10
USE_TPU = False
BATCH_SIZE = 1


tokenizer = tokenization.BertTokenizer(
      vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)
#tokenize
line = tokenization.convert_to_unicode(SENT_A)
line = line.strip()
tokens_a = tokenizer.tokenize(line)
line = tokenization.convert_to_unicode(SENT_B)
line = line.strip()
tokens_b = tokenizer.tokenize(line)
print (tokens_a)
print (tokens_b)

# generate token with mask and segment_ids
tokens = []
segment_ids = []
tokens.append("[CLS]")
segment_ids.append(0)
for token in tokens_a:
  tokens.append(token)
  segment_ids.append(0)
tokens.append("[SEP]")
segment_ids.append(0)
for token in tokens_b:
  tokens.append(token)
  segment_ids.append(1)
tokens.append("[SEP]")
segment_ids.append(1)
(tokens_all, masked_lm_positions, masked_lm_labels, segment_ids) = create_masked_lm_predictions_based_given(
             tokens, MAX_PREDICTIONS_PER_SEQ, segment_ids)

# print tokens
# print len(tokens)
# print masked_lm_positions
# print masked_lm_labels
# generate instance
instance = TrainingInstance(
            tokens=tokens_all,
            segment_ids=segment_ids,
            is_random_next=IS_RANDOM_NEXT,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)

# generate tf_example
features = generate_example_given_instance(instance, tokenizer, MAX_SEQ_LENGTH, MAX_PREDICTIONS_PER_SEQ)

print (features)

bert_config = modeling.BertConfig(BERT_CONFIG_FILE)
device = torch.device("cpu")
model = modeling.BertForPreTraining(bert_config)
model.load_state_dict(torch.load(INIT_CHECKPOINT_PT, map_location='cpu'))
# model.bert.from_pretrained(INIT_DIRECTORY)
model.to(device)


print ('loaded model')

#TODO resolve features
all_input_ids = torch.tensor([features['input_ids']], dtype=torch.long)
all_input_mask = torch.tensor([features['input_mask']], dtype=torch.long)
all_input_type_ids = torch.tensor([features['segment_ids']], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_input_type_ids, all_example_index)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

model.eval()



for input_ids, input_mask, input_type_ids, example_indices in eval_dataloader:
    # print(input_ids)
    # print(input_mask)
    # print(example_indices)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)

    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
    print (masked_lm_logits_scores.shape)
    print (seq_relationship_logits)

    masked_lm_predictions = torch.nn.functional.log_softmax(masked_lm_logits_scores[:,features['masked_lm_positions'],:], dim=-1)
    masked_lm_predictions = torch.argmax(masked_lm_predictions, dim=-1)
    next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_logits, dim=-1)
    next_sentence_predictions = torch.argmax(next_sentence_log_probs, dim=-1)
    print (masked_lm_predictions)
    print (next_sentence_log_probs)
    print (next_sentence_predictions)


    input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'])
    pred_tokens = tokenizer.convert_ids_to_tokens(masked_lm_predictions.detach().numpy()[0].astype(int))
    true_tokens = tokenizer.convert_ids_to_tokens(features['masked_lm_ids'])
    mask_count = 0
    print (features['masked_lm_positions'])
    for i in range(len(input_tokens)):
        if i == features['masked_lm_positions'][mask_count]:
            print("[%s: %s (%f)] " % (true_tokens[mask_count], pred_tokens[mask_count], features['masked_lm_weights'][mask_count])),
            # print("[%s: %s (%f)] " % (true_tokens[mask_count], pred_tokens[mask_count], features['masked_lm_weights'][mask_count]),end='')
            mask_count += 1
        else:
            print("%s " % (input_tokens[i])),
            # print("%s " % (input_tokens[i]),end='')
    print("\n")
    print("true: %d, pred: %d\n" % (features['next_sentence_labels'][0], next_sentence_predictions.detach().numpy()[0]))
    print(np.exp(next_sentence_log_probs.detach().numpy()))