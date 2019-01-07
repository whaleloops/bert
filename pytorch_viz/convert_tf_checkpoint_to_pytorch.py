# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import argparse

import numpy as np



def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)

def convert_tf_checkpoint_to_tmp(tf_checkpoint_path, bert_config_file):
    import tensorflow as tf
    import pickle
    
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
    # Load weights from TF model TODO
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    with open("tmp_names", "wb") as fp:   #Pickling
      pickle.dump(names, fp)
    with open("tmp_arrays", "wb") as fp:   #Pickling
      pickle.dump(arrays, fp)


def convert_tmp_to_pytorch(bert_config_file, pytorch_dump_path):
    import torch
    from modeling import BertConfig, BertForPreTraining
    import pickle

    with open("tmp_names", "rb") as fp:   # Unpickling
      # names = pickle.load(fp, encoding='iso-8859-1')
      names = pickle.load(fp)
    with open("tmp_arrays", "rb") as fp:   # Unpickling
      # arrays = pickle.load(fp, encoding='iso-8859-1')
      arrays = pickle.load(fp)

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if name[-1] in ["adam_v", "adam_m", 'global_step']:
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if fullmatch(r'[A-Za-z]+_\d+', m_name):
            # if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--mode",
                        default = None,
                        type = int,
                        required = True,
                        help = "which step to take")
    args = parser.parse_args()
    if args.mode==0:
        convert_tf_checkpoint_to_tmp(args.tf_checkpoint_path,
                                     args.bert_config_file)
    else:
        convert_tmp_to_pytorch(args.bert_config_file,
                                     args.pytorch_dump_path)