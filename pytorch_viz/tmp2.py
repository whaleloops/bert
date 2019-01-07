# -*- coding: iso-8859-15 -*-

import collections
import random

import modeling, tokenization

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

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



SENT_A = "Nancy has just got a job as a secretary in a company. Monday was the first day she \
went to work, so she was very excited and arrived early. "
# SENT_B =  "She pushed the door open and found nobody there. \"I am the  first to arrive.\" She \
# thought and come to her desk. She was surprised to find a bunch of flowers on it. "
SENT_B =  "There is a philosophic pleasure in opening one's treasures to the modest young. "
IS_RANDOM_NEXT = False

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
torch.manual_seed(123)

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

# load model
bert_config = modeling.BertConfig(BERT_CONFIG_FILE)
device = torch.device("cpu")
model = modeling.BertForPreTraining(bert_config)
model.load_state_dict(torch.load(INIT_CHECKPOINT_PT, map_location='cpu'))
model_out = modeling.BertWithOutEmbedding(model)
# model.bert.from_pretrained(INIT_DIRECTORY)
# model.to(device)
model_out.to(device)

print ('loaded model')

child_counter = 0
for child in model.children():
    print(" child", child_counter, "is -")
    print(child)
    child_counter += 1

#resolve features
input_ids = torch.tensor([features['input_ids']], dtype=torch.long)
input_mask = torch.tensor([features['input_mask']], dtype=torch.long)
input_type_ids = torch.tensor([features['segment_ids']], dtype=torch.long)
# masked_lm_labels = torch.tensor([features['masked_lm_labels']], dtype=torch.long)
next_sentence_labels = torch.tensor([features['next_sentence_labels']], dtype=torch.long)
masked_lm_ids = torch.tensor([features['masked_lm_ids']], dtype=torch.long)
masked_lm_weights = torch.tensor([features['masked_lm_weights']], dtype=torch.long)
masked_lm_positions = torch.tensor([features['masked_lm_positions']], dtype=torch.long)
example_index = torch.arange(input_ids.size(0), dtype=torch.long)

print features['input_ids']
print 'sentence seperated between :'
idxs = np.where(np.array(features['input_ids']) == 102)[0]
print idxs
masked_lm_labels = torch.tensor([features['input_ids']], dtype=torch.long)
print masked_lm_labels
masked_lm_labels[masked_lm_labels==0] = -1.0
# masked_lm_labels[0,0:idxs[0]+1] = -1.0
masked_lm_labels[0,0] = -1.0
masked_lm_labels[0,idxs[0]+1:idxs[1]+1] = -1.0
print masked_lm_labels

# i1 = 0
# # for param in model1.parameters():
# for name, param in model.named_parameters():
#   # param.requires_grad = False
#   if param.requires_grad:
    
#     if name=='bert.embeddings.word_embeddings.weight':
#       print name
#       param.requires_grad=False
#     # print name, param.data
#   i1 += 1
# print i1


# run eval first time
masked_lm_loss, next_sentence_loss, masked_lm_logits_scores, seq_relationship_logits, _ = model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask,
                                    masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels, is_id=True)
# masked_lm_logits_scores #(1, 128, 30522)
# seq_relationship_logits #(1, 2)
print masked_lm_loss.item()
print next_sentence_loss.item()
masked_lm_predictions = torch.nn.functional.log_softmax(masked_lm_logits_scores, dim=-1)
masked_lm_predictions = torch.argmax(masked_lm_predictions, dim=-1)
next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_logits, dim=-1)
next_sentence_predictions = torch.argmax(next_sentence_log_probs, dim=-1)
print (masked_lm_predictions)
input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'])
pred_tokens = tokenizer.convert_ids_to_tokens(masked_lm_predictions.detach().numpy()[0].astype(int))
mask_count = 0
for i in range(len(input_tokens)):
  print("%s " % (pred_tokens[i])),
  # print("%s " % (pred_tokens[i]),end='')
print("\n")
print("true: %d, pred: %d\n" % (features['next_sentence_labels'][0], next_sentence_predictions.detach().numpy()[0]))
print(np.exp(next_sentence_log_probs.detach().numpy()))

embeddings = torch.tensor(model.bert.embeddings.word_embeddings_value.detach().numpy(), dtype=torch.float)
del model

def run_ADAM_transfer(embeddings, input_type_ids, input_mask, input_ids, next_sentence_labels, 
                      model_out, num_steps=20,
                      style_weight=1, content_weight=1):
  # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
  # optimizer = torch.optim.Adam([model.bert.embeddings.word_embeddings.weight.requires_grad_()], lr=2e-4, weight_decay=1e-5)
  optimizer = torch.optim.Adam([embeddings.requires_grad_()], lr=5e-3, weight_decay=1e-5)
  # optimizer = torch.optim.LBFGS([input_ids.requires_grad_()])
  losses = np.zeros((num_steps))
  for t in range(num_steps):
    masked_lm_loss, next_sentence_loss, masked_lm_logits_scores, seq_relationship_logits, sequence_output = model_out(embeddings, token_type_ids=input_type_ids, attention_mask=input_mask,
                                        masked_lm_labels=masked_lm_labels , next_sentence_label=next_sentence_labels)
    masked_lm_loss *= style_weight
    next_sentence_loss *= content_weight
    loss = masked_lm_loss+next_sentence_loss
    losses[t] = loss.item()
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the Tensors it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()
    if t % 5 == 0:
      print("------")
      print("run {}:".format(t))
      print('masked_lm_loss: {:4f} next_sentence_loss: {}'.format(
          masked_lm_loss.item(), next_sentence_loss.item()))
      print("-----")
      print(embeddings)
      masked_lm_predictions = torch.nn.functional.log_softmax(masked_lm_logits_scores, dim=-1)
      masked_lm_predictions = torch.argmax(masked_lm_predictions, dim=-1)
      next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_logits, dim=-1)
      next_sentence_predictions = torch.argmax(next_sentence_log_probs, dim=-1)
      print (masked_lm_predictions)
      input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'])
      pred_tokens = tokenizer.convert_ids_to_tokens(masked_lm_predictions.detach().numpy()[0].astype(int))
      true_tokens = tokenizer.convert_ids_to_tokens(features['masked_lm_ids'])
      mask_count = 0
      for i in range(len(input_tokens)):
        print("%s " % (pred_tokens[i])),
        # print("%s " % (input_tokens[i]),end='')
      print("\n")
      print("true: %d, pred: %d\n" % (features['next_sentence_labels'][0], next_sentence_predictions.detach().numpy()[0]))
      print(np.exp(next_sentence_log_probs.detach().numpy()))
  #plot losses

  fig, ax = plt.subplots()
  ax.plot(losses)
  ax.set(xlabel='number of steps', ylabel='losses',
     title='losses vs. number of steps')
  ax.set_ylim([None,30])
  plt.savefig('loss.png')
  plt.show()
  return masked_lm_logits_scores, seq_relationship_logits

def run_LBFGS_transfer(embeddings, input_type_ids, input_mask, input_ids, next_sentence_labels, 
                      model_out, num_steps=20,
                      style_weight=1, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    masked_lm_loss, next_sentence_loss, masked_lm_logits_scores, seq_relationship_logits, sequence_output = model_out(embeddings, token_type_ids=input_type_ids, attention_mask=input_mask,
                                      masked_lm_labels=masked_lm_labels , next_sentence_label=next_sentence_labels)
    optimizer = torch.optim.LBFGS([embeddings.requires_grad_()])
    print('Optimizing.. total %d steps' % num_steps)
    run = [0]
    while run[0] <= num_steps:
      def closure():
        # correct the values of updated input image
        # input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        masked_lm_loss, next_sentence_loss, masked_lm_logits_scores, seq_relationship_logits, sequence_output = model_out(embeddings, token_type_ids=input_type_ids, attention_mask=input_mask,
                                      masked_lm_labels=masked_lm_labels , next_sentence_label=next_sentence_labels)


        masked_lm_loss *= style_weight
        next_sentence_loss *= content_weight

        loss = masked_lm_loss + next_sentence_loss
        loss.backward()
        # next_sentence_loss.backward()

        run[0] += 1
        if run[0] % 5 == 0:
          print("run {}:".format(run))
          print('masked_lm_loss: {:4f} next_sentence_loss: {}'.format(
              masked_lm_loss.item(), next_sentence_loss.item()))
          print(embeddings)

          masked_lm_predictions = torch.nn.functional.log_softmax(masked_lm_logits_scores, dim=-1)
          masked_lm_predictions = torch.argmax(masked_lm_predictions, dim=-1)
          next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_logits, dim=-1)
          next_sentence_predictions = torch.argmax(next_sentence_log_probs, dim=-1)
          print (masked_lm_predictions)
          input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'])
          pred_tokens = tokenizer.convert_ids_to_tokens(masked_lm_predictions.detach().numpy()[0].astype(int))
          true_tokens = tokenizer.convert_ids_to_tokens(features['masked_lm_ids'])
          mask_count = 0
          for i in range(len(input_tokens)):
            print("%s " % (pred_tokens[i])),
            # print("%s " % (input_tokens[i]),end='')
          print("\n")
          print("true: %d, pred: %d\n" % (features['next_sentence_labels'][0], next_sentence_predictions.detach().numpy()[0]))
          print(np.exp(next_sentence_log_probs.detach().numpy()))

        return masked_lm_loss + next_sentence_loss
        # return next_sentence_loss

      optimizer.step(closure)

    # a last correction...
    # input_img.data.clamp_(0, 1)

    return masked_lm_logits_scores, seq_relationship_logits

masked_lm_logits_scores, seq_relationship_logits = run_ADAM_transfer(embeddings, input_type_ids, input_mask, input_ids, next_sentence_labels, 
                      model_out, num_steps=61,
                      style_weight=1, content_weight=1)

masked_lm_predictions = torch.nn.functional.log_softmax(masked_lm_logits_scores, dim=-1)
masked_lm_predictions = torch.argmax(masked_lm_predictions, dim=-1)
next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_logits, dim=-1)
next_sentence_predictions = torch.argmax(next_sentence_log_probs, dim=-1)
print (masked_lm_predictions)

input_tokens = tokenizer.convert_ids_to_tokens(features['input_ids'])
pred_tokens = tokenizer.convert_ids_to_tokens(masked_lm_predictions.detach().numpy()[0].astype(int))
true_tokens = tokenizer.convert_ids_to_tokens(features['masked_lm_ids'])
mask_count = 0
for i in range(len(input_tokens)):
  print("%s " % (pred_tokens[i])),
  # print("%s " % (input_tokens[i]),end='')
print("\n")
print("true: %d, pred: %d\n" % (features['next_sentence_labels'][0], next_sentence_predictions.detach().numpy()[0]))
print(np.exp(next_sentence_log_probs.detach().numpy()))