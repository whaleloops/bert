# -*- coding: iso-8859-15 -*-

import collections
import random
import pickle

import modeling, tokenization

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, masked_lm_positions, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
  """Computes the loss and accuracy of the model."""
  # masked_lm_log_probs = np.reshape(masked_lm_log_probs,
  #                                  [-1, masked_lm_log_probs.shape[-1]]) #(number of masks, vocab_size)
  masked_lm_predictionss = np.argmax(masked_lm_log_probs, axis=-1)
  masked_lm_predictions = np.zeros_like(masked_lm_positions)
  for i in range(masked_lm_predictionss.shape[0]):
    masked_lm_predictions[i] = masked_lm_predictionss[i][masked_lm_positions[i]]

  masked_lm_predictions = np.reshape(masked_lm_predictions, [-1])
  masked_lm_ids = np.reshape(masked_lm_ids, [-1])
  masked_lm_weights = np.reshape(masked_lm_weights, [-1])
  masked_lm_accuracy = np.average(masked_lm_ids==masked_lm_predictions, weights=masked_lm_weights)
  masked_lm_mean_loss = masked_lm_example_loss
  next_sentence_log_probs = np.reshape(
      next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
  next_sentence_predictions = np.argmax(
      next_sentence_log_probs, axis=-1)
  next_sentence_labels = np.reshape(next_sentence_labels, [-1])
  next_sentence_accuracy = np.mean(next_sentence_labels==next_sentence_predictions)
  next_sentence_mean_loss = next_sentence_example_loss
  return {
      "masked_lm_accuracy": masked_lm_accuracy,
      "masked_lm_loss": masked_lm_mean_loss,
      "next_sentence_accuracy": next_sentence_accuracy,
      "next_sentence_loss": next_sentence_mean_loss,
  }

if __name__ == "__main__":
  VOCAB_FILE = '../../uncased_L-12_H-768_A-12/vocab.txt'
  BERT_CONFIG_FILE = '../../uncased_L-12_H-768_A-12/bert_config.json'
  INIT_DIRECTORY = "../../uncased_L-12_H-768_A-12"
  INIT_CHECKPOINT_PT = "../../uncased_L-12_H-768_A-12/pytorch_model.bin"
  RANDOM_SEED = 12345
  MAX_PREDICTIONS_PER_SEQ = 20
  MAX_SEQ_LENGTH = 128
  DO_LOWER_CASE = True


  # LEARNING_RATE = 2e-5
  # NUM_TRAIN_STEPS = 1
  # NUM_WARMUP_STEPS = 10
  # USE_TPU = False
  # BATCH_SIZE = 1


  # load model
  bert_config = modeling.BertConfig(BERT_CONFIG_FILE)
  device = torch.device("cpu")
  model = modeling.BertForPreTraining(bert_config)
  model.load_state_dict(torch.load(INIT_CHECKPOINT_PT, map_location='cpu'))
  # model.bert.from_pretrained(INIT_DIRECTORY)
  model.to(device)
  print ('model loaded')




  #resolve features
  with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

  print ("%d total samples" % len(features))

  all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
  all_input_type_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
  all_masked_lm_labels = torch.tensor([f['masked_lm_labels'] for f in features], dtype=torch.long)
  all_next_sentence_labels = torch.tensor([f['next_sentence_labels'] for f in features], dtype=torch.long)
  all_masked_lm_ids = torch.tensor([f['masked_lm_ids'] for f in features], dtype=torch.long)
  all_masked_lm_weights = torch.tensor([f['masked_lm_weights'] for f in features], dtype=torch.long)
  all_masked_lm_positions = torch.tensor([f['masked_lm_positions'] for f in features], dtype=torch.long)
  all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

  eval_data = TensorDataset(all_input_ids, all_input_mask, all_input_type_ids, all_masked_lm_labels, 
    all_next_sentence_labels, all_masked_lm_ids, all_masked_lm_weights, all_masked_lm_positions, 
    all_example_index)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=30)
  model.eval()

  for input_ids, input_mask, input_type_ids, masked_lm_labels, next_sentence_labels, masked_lm_ids, masked_lm_weights, masked_lm_positions, example_indices in eval_dataloader:
    masked_lm_loss, next_sentence_loss, prediction_scores, seq_relationship_score = model(input_ids, token_type_ids=input_type_ids, attention_mask=input_mask,
                                               masked_lm_labels=masked_lm_labels , next_sentence_label=next_sentence_labels)
    # print (masked_lm_loss)
    # print (next_sentence_loss)
    masked_lm_log_probs =  torch.nn.functional.log_softmax(prediction_scores, dim=-1).detach().numpy()
    next_sentence_log_probs = torch.nn.functional.log_softmax(seq_relationship_score, dim=-1).detach().numpy()
    masked_lm_example_loss = masked_lm_loss.detach().numpy()
    next_sentence_example_loss = next_sentence_loss.detach().numpy()
    masked_lm_ids        = masked_lm_ids.detach().numpy()
    next_sentence_labels = next_sentence_labels.detach().numpy()
    masked_lm_weights    = masked_lm_weights.detach().numpy()
    masked_lm_positions  = masked_lm_positions.detach().numpy()

    evals = metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, masked_lm_positions, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels)
    print (evals)


