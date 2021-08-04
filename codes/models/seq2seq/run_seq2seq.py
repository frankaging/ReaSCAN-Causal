#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

from decode_graphical_models import *
from seq2seq.ReaSCAN_dataset import *


# In[ ]:


def arg_parse():
    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

    # General arguments
    parser.add_argument("--mode", type=str, default="run_tests", help="train, test or predict", required=True)
    parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                               "saved. Will be created if doesn't exist.")
    parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")

    # Data arguments
    parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
    parser.add_argument("--data_directory", type=str, default="data/uniform_dataset", help="Path to folder with data.")
    parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                        help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                        help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                        help="Whether to generate vocabularies based on the data.")
    parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                        help="Whether to use previously saved vocabularies.")

    # Training and learning arguments
    parser.add_argument("--training_batch_size", type=int, default=50)
    parser.add_argument("--k", type=int, default=0, help="How many examples from the few-shot split to move to train.")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
    parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_decay_steps', type=float, default=20000)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                         "test set (without teacher forcing).")
    parser.add_argument("--max_training_iterations", type=int, default=100000)
    parser.add_argument("--weight_target_loss", type=float, default=0.3, help="Only used if --auxiliary_task set.")

    # Testing and predicting arguments
    parser.add_argument("--max_testing_examples", type=int, default=None)
    parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
    parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
                                                                           "is stopped regardless of whether an EOS token "
                                                                           "was generated.")
    parser.add_argument("--output_file_name", type=str, default="predict.json")

    # Situation Encoder arguments
    parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                        action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                                  "For more information, read grounded SCAN documentation.")
    parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                        action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                                   "For more information, read grounded SCAN documentation.")
    parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
    parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                       "model.")
    parser.add_argument("--cnn_dropout_p", type=float, default=0.1, help="Dropout applied to the output features of the "
                                                                         "world state model.")
    parser.add_argument("--auxiliary_task", dest="auxiliary_task", default=False, action="store_true",
                        help="If set to true, the model predicts the target location from the joint attention over the "
                             "input instruction and world state.")
    parser.add_argument("--no_auxiliary_task", dest="auxiliary_task", default=True, action="store_false")

    # Command Encoder arguments
    parser.add_argument("--embedding_dimension", type=int, default=25)
    parser.add_argument("--num_encoder_layers", type=int, default=1)
    parser.add_argument("--encoder_hidden_size", type=int, default=100)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")
    parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
    parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

    # Decoder arguments
    parser.add_argument("--num_decoder_layers", type=int, default=1)
    parser.add_argument("--attention_type", type=str, default='bahdanau', choices=['bahdanau', 'luong'],
                        help="Luong not properly implemented.")
    parser.add_argument("--decoder_dropout_p", type=float, default=0.3, help="Dropout on decoder embedding and LSTM.")
    parser.add_argument("--decoder_hidden_size", type=int, default=100)
    parser.add_argument("--conditional_attention", dest="conditional_attention", default=True, action="store_true",
                        help="If set to true joint attention over the world state conditioned on the input instruction is"
                             " used.")
    parser.add_argument("--no_conditional_attention", dest="conditional_attention", default=False, action="store_false")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args


# In[ ]:


if __name__ == "__main__":
    
    # Loading arguments
    args = arg_parse()
    try:        
        get_ipython().run_line_magic('matplotlib', 'inline')
        is_jupyter = True
    except:
        is_jupyter = False
    print(args)

