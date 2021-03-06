# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np

import tokenization
import tensorflow as tf


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


def create_multi_features_from_instances(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):

  total_written = 0
  features = []
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    # print (instance.tokens)
    # print (instance.segment_ids)

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

    feature = collections.OrderedDict()
    feature["input_ids"] = input_ids
    feature["input_mask"] = input_mask
    feature["segment_ids"] = segment_ids
    feature["masked_lm_positions"] = masked_lm_positions
    feature["masked_lm_ids"] = masked_lm_ids
    feature["masked_lm_weights"] = masked_lm_weights
    feature["next_sentence_labels"] = [next_sentence_label]
    feature['masked_lm_labels'] = np.ones_like(input_ids)*-1
    feature['masked_lm_labels'][masked_lm_positions] = masked_lm_ids
    features.append(feature)

    total_written += 1

  print ("Wrote %d total instances pt" % total_written)
  return features

def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, mask_given=False):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    # with open(input_file, "r", encoding="iso-8859-15") as reader:
    with open(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      tmp = create_biinstances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng, mask_given=mask_given)
      instances.extend(tmp)

  rng.shuffle(instances)
  return instances


def create_biinstances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, mask_given=False):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

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


        if mask_given:
          rng2 = random.Random(RANDOM_SEED)
          (tokens, masked_lm_positions, masked_lm_labels, segment_ids) = create_masked_lm_predictions_based_given(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, segment_ids, rng2)
        else:
          (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)        
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  masked_lm = collections.namedtuple("masked_lm", ["index", "label"])  # pylint: disable=invalid-name

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.1:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(masked_lm(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  # print (tokens)
  # print (output_tokens)
  # print (masked_lm_positions)
  # print (masked_lm_labels)
  # abc

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions_based_given(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, segment_ids, rng2):
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
      masked_lm_labels.append(tokens[i])
      segment_ids_new.append(segment_ids[i])
      idx+=1
    else:
      if rng2.random() < 0.05:
        tok = vocab_words[rng2.randint(0, len(vocab_words) - 1)]
      output_tokens.append(tok)
      segment_ids_new.append(segment_ids[i])
      idx+=1
    i+=1

  # print (tokens)
  # print (output_tokens)
  # print (masked_lm_positions)
  # print (masked_lm_labels)
  # abc

  return (output_tokens, masked_lm_positions, masked_lm_labels, segment_ids_new)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    # print (instance.tokens)
    # print (instance.segment_ids)

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
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    # if inst_index < 20:
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in instance.tokens]))

    #   for feature_name in features.keys():
    #     feature = features[feature_name]
    #     values = []
    #     if feature.int64_list.value:
    #       values = feature.int64_list.value
    #     elif feature.float_list.value:
    #       values = feature.float_list.value
    #     tf.logging.info(
    #         "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  print ("Wrote %d total instances tf" % total_written)

def main():

  tokenizer = tokenization.BertTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

  input_files = [INPUT_FILE]

  rng = random.Random(RANDOM_SEED)
  instances = create_training_instances(
      input_files, tokenizer, MAX_SEQ_LENGTH, DUPE_FACTOR,
      SHORT_SEQ_PROB, MASKED_LM_PROB, MAX_PREDICTIONS_PER_SEQ,
      rng, mask_given=MASK_GIVEN)

  import pickle
  with open('pytorch/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

  features = create_multi_features_from_instances(instances, tokenizer, MAX_SEQ_LENGTH,
                                  MAX_PREDICTIONS_PER_SEQ, 'output_files')

  with open('pytorch/clinic_train.pkl', 'wb') as f:
    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)

  write_instance_to_example_files(instances, tokenizer, MAX_SEQ_LENGTH,
                                    MAX_PREDICTIONS_PER_SEQ, output_files)


if __name__ == "__main__":
  # rng2.ran
  INPUT_FILE= './clinic_train.txt'
  # INPUT_FILE= './sample_text.txt'
  VOCAB_FILE = '../uncased_L-12_H-768_A-12/vocab6.txt'
  output_files = ['../tmp/clinic_train3.tfrecord']

  RANDOM_SEED = 12345
  MAX_PREDICTIONS_PER_SEQ = 20
  MAX_SEQ_LENGTH = 128
  DO_LOWER_CASE = True

  DUPE_FACTOR=5
  SHORT_SEQ_PROB=0.1
  MASKED_LM_PROB=0.15
  MASK_GIVEN=False

  main()
