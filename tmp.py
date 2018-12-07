# -*- coding: iso-8859-15 -*-

import tensorflow as tf
import numpy as np

import collections
import random

import tokenization
from create_pretraining_data import TrainingInstance

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
    print 'too many masks'
  # print (tokens)
  # print (output_tokens)
  # print (masked_lm_positions)
  # print (masked_lm_labels)
  # abc

  return (output_tokens, masked_lm_positions, masked_lm_labels, segment_ids_new)

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

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
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
  features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
  features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
  features["next_sentence_labels"] = create_int_feature([next_sentence_label])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

def pred_input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    d = tf.data.TFRecordDataset(input_files)
    d = d.repeat(1)

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    return d

  return input_fn

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        
        all_attention_weight = model.all_attention_weight

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(
               tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:

            q,w = masked_lm_ids.shape
            masked_lm_log_probss = tf.reshape(masked_lm_log_probs,
                                               [q,w,-1]) #(batch, number of masks pre batch, vocab_size)
            masked_lm_predictions = tf.argmax(
                  masked_lm_log_probss, axis=-1, output_type=tf.int32)
            next_sentence_log_probss = tf.reshape(
                  next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
            next_sentence_predictions = tf.argmax(
                  next_sentence_log_probss, axis=-1, output_type=tf.int32)

            predicts = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_ids': masked_lm_ids,
                'masked_lm_weights': masked_lm_weights,
                'next_sentence_labels': next_sentence_labels,
                'masked_lm_predictions': masked_lm_predictions,
                'masked_lm_log_probs': masked_lm_log_probss,
                'next_sentence_log_probs': next_sentence_log_probs,
                'next_sentence_predictions': next_sentence_predictions,
                'all_attention_weight': all_attention_weight,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode, 
              predictions=predicts,
              scaffold_fn=scaffold_fn)
        else:
              raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


# SENT_A = "Indeed, it was recorded in Blazing Star that a fortunate early 力 riser had once picked up on the \
# highway a solid chunk of gold quartz which the rain had freed from its incumbering soil, and washed into \
# immediate and glittering popularity."
# SENT_B =  "Possibly this may have been the reason why early risers in that locality, during the rainy \
# season, adopted a thoughtful habit of body, and seldom lifted their eyes to the rifted or \
# india-ink washed skies 力 above them."
SENT_A = "William Shakespeare was an English poet, playwright and actor, widely regarded as the \
greatest writer in the English language and the world's greatest dramatist."
SENT_B =  "He is often called England's national poet and the Bard of Avon"
IS_RANDOM_NEXT = True



VOCAB_FILE = '../uncased_L-12_H-768_A-12/vocab.txt'
DO_LOWER_CASE = True
MAX_PREDICTIONS_PER_SEQ = 20
MAX_SEQ_LENGTH = 128
RECORD_FILE = '../tmp/pred.tfrecord'
BERT_CONFIG_FILE = '../uncased_L-12_H-768_A-12/bert_config.json'
OUTPUT_DIR = '../tmp/pretraining_output'
INIT_CHECKPOINT = '../uncased_L-12_H-768_A-12/bert_model.ckpt'
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
print tokens_a
print tokens_b

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
tf_example = generate_example_given_instance(instance, tokenizer, MAX_SEQ_LENGTH, MAX_PREDICTIONS_PER_SEQ)

print tf_example

# "example" is of type tf.train.Example.
with tf.python_io.TFRecordWriter('../tmp/pred.tfrecord') as writer:
  writer.write(tf_example.SerializeToString())

import modeling
# tf.logging.set_verbosity(tf.logging.INFO)

bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)

tf.gfile.MakeDirs(OUTPUT_DIR)

input_files = []
for input_pattern in RECORD_FILE.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

tf.logging.info("*** Input Files ***")
num_records = 0
for input_file in input_files:
    tf.logging.info("  %s" % input_file)
    for fn in input_files:
        for record in tf.python_io.tf_record_iterator(fn):
            num_records += 1
tf.logging.info ("  total number of records: %d" % num_records)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
      cluster=None,
      master=None,
      model_dir=OUTPUT_DIR,
      save_checkpoints_steps=1000,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=1000,
          num_shards=8,
          per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      learning_rate=LEARNING_RATE, 
      num_train_steps=NUM_TRAIN_STEPS, 
      num_warmup_steps=NUM_WARMUP_STEPS, 
      use_tpu=USE_TPU, 
      use_one_hot_embeddings=USE_TPU)

# If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=USE_TPU,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=BATCH_SIZE,
      eval_batch_size=BATCH_SIZE,
      predict_batch_size=BATCH_SIZE)

predict_input_fn = pred_input_fn_builder(
        input_files=input_files,
        max_seq_length=MAX_SEQ_LENGTH,
        max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ)


for item in estimator.predict(input_fn=predict_input_fn):         
    input_tokens = tokenizer.convert_ids_to_tokens(item['input_ids'])
    pred_tokens = tokenizer.convert_ids_to_tokens(item['masked_lm_predictions'])
    true_tokens = tokenizer.convert_ids_to_tokens(item['masked_lm_ids'])
    mask_count = 0
    print item['masked_lm_positions']
    for i in range(len(input_tokens)):
        if i == item['masked_lm_positions'][mask_count]:
            print("[%s: %s (%f)] " % (true_tokens[mask_count], pred_tokens[mask_count], item['masked_lm_weights'][mask_count])),
            mask_count += 1
        else:
            print("%s " % (input_tokens[i])),
    print("\n"),
    print("true: %d, pred: %d\n" % (item['next_sentence_labels'], item['next_sentence_predictions'])),
    print(np.exp(item['next_sentence_log_probs']))
    print item['all_attention_weight'].shape
    enc_atts = item['all_attention_weight']
                        
