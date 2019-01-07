# -*- coding: iso-8859-15 -*-

import collections
import random
import pickle

import modeling, tokenization

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


if __name__ == "__main__":
  VOCAB_FILE = '../../uncased_L-12_H-768_A-12/model/model07/vocab.txt'
  BERT_CONFIG_FILE = '../../uncased_L-12_H-768_A-12/model/model07/bert_config.json'
  INIT_DIRECTORY = "../../uncased_L-12_H-768_A-12"
  INIT_CHECKPOINT_PT = "../../uncased_L-12_H-768_A-12/model/model07/bert7.pth.tar"
  INPUT_FILE = "drop_0_test.pkl"
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
  model1 = modeling.BertForPreTraining(bert_config)
  # model2 = modeling.BertForPreTraining(bert_config)
  print (model1.bert.embeddings.word_embeddings.weight)
  print (model1.bert.encoder.layer[1])

  model1.load_state_dict(torch.load(INIT_CHECKPOINT_PT, map_location='cpu'))
  # model1.bert.from_pretrained(INIT_DIRECTORY)
  model1.to(device)
  print (model1.bert.embeddings.word_embeddings.weight)
  print ('model loaded')

  child_counter = 0
  for child in model1.children():
      print(" child", child_counter, "is -")
      print(child)
      child_counter += 1

