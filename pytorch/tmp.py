# -*- coding: iso-8859-15 -*-

import collections
import random

import modeling, tokenization
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
    if tok==u'\u529b':
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

  return features


SENT_A = "Indeed, it was recorded in Blazing Star that a fortunate early 力 riser had once picked up on the \
highway a solid chunk of gold quartz which the rain had freed from its incumbering soil, and washed into \
immediate and glittering popularity."
SENT_B =  "Possibly this may have been the reason why early risers in that locality, during the rainy \
season, adopted a thoughtful habit of body, and seldom lifted their eyes to the rifted or \
india-ink washed skies 力 above them."
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

tokenizer = tokenization.FullTokenizer(
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
all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
all_input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_input_type_ids, all_example_index)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

model.eval()



for input_ids, input_mask, input_type_ids, example_indices in eval_dataloader:
    print(input_ids)
    print(input_mask)
    print(example_indices)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)

    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

    print (masked_lm_logits_scores.shape)
    print (seq_relationship_logits.shape)
      # q,w = masked_lm_ids.shape
      # masked_lm_log_probss = tf.reshape(masked_lm_logits_scores,
      #                                    [q,w,-1]) #(batch, number of masks pre batch, vocab_size)
      # masked_lm_predictions = tf.argmax(
      #       masked_lm_log_probss, axis=-1, output_type=tf.int32)
      # next_sentence_log_probss = tf.reshape(
      #       next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
      # next_sentence_predictions = tf.argmax(
      #       next_sentence_log_probss, axis=-1, output_type=tf.int32)