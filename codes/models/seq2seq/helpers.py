import torch
import numpy as np
from typing import List
import logging
import argparse

logger = logging.getLogger(__name__)

def sequence_mask(sequence_lengths: torch.LongTensor, max_len=None) -> torch.tensor:
    """
    Create a sequence mask that masks out all indices larger than some sequence length as defined by
    sequence_lengths entries.

    :param sequence_lengths: [batch_size] sequence lengths per example in batch
    :param max_len: int defining the maximum sequence length in the batch
    :return: [batch_size, max_len] boolean mask
    """
    if max_len is None:
        max_len = sequence_lengths.data.max()
    batch_size = sequence_lengths.size(0)
    sequence_range = torch.arange(0, max_len).long().to(device=sequence_lengths.device)

    # [batch_size, max_len]
    sequence_range_expand = sequence_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(sequence_range_expand))

    # [batch_size, max_len](boolean array of which elements to include)
    return sequence_range_expand < seq_length_expand


def log_parameters(model: torch.nn.Module) -> {}:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Total parameters: %d" % n_params)
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("%s : %s" % (name, list(p.size())))


def sequence_accuracy(prediction: List[int], target: List[int]) -> float:
    # print("****", prediction)
    # print("****", target)
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        difference = len(prediction) - len(target)
        target.extend([-1] * difference)
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100

def arg_parse():
    parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

    # General arguments
    parser.add_argument("--mode", type=str, default="train", help="train, test or predict")
    parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                               "saved. Will be created if doesn't exist.")
    parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")

    # Data arguments
    parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
    parser.add_argument("--data_directory", type=str, default="../../../data-files/ReaSCAN-Simple/", help="Path to folder with data.")
    parser.add_argument("--input_vocab_path", type=str, default="input_vocabulary.txt",
                        help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
    parser.add_argument("--target_vocab_path", type=str, default="target_vocabulary.txt",
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
    
    # Counterfactual arguments
    parser.add_argument("--cf_mode", type=str, default="random", help="What mode are you in for your counterfactual training.")
    parser.add_argument("--run_name", type=str, default="seq2seq")
    parser.add_argument("--cf_sample_p", type=float, default=0.25, help="Percentage of examples in a batch to include counterfactual loss")
    parser.add_argument("--checkpoint_save_every", type=int, default=2000)
    parser.add_argument("--evaluate_checkpoint", type=str, default="")
    parser.add_argument("--include_task_loss", dest="include_task_loss", default=False,
                        action="store_true", help="Whether to include task loss during counterfactual training.")
    parser.add_argument("--include_cf_loss", dest="include_cf_loss", default=False,
                        action="store_true", help="Whether to include counterfactual loss during counterfactual training")
    parser.add_argument("--cf_loss_weight", type=float, default=1.0, help="Weight of cf loss comparing to the task loss")
    parser.add_argument("--is_wandb", dest="is_wandb", default=False,
                        action="store_true", help="Whether to report metrics to weights and bias.")
    parser.add_argument("--intervene_attribute", type=int, default=-1)
    parser.add_argument("--intervene_time", type=int, default=-1)
    parser.add_argument("--intervene_dimension_size", type=int, default=25)
    parser.add_argument("--include_cf_auxiliary_loss", dest="include_cf_auxiliary_loss", default=False, action="store_true",
                        help="If set to true, the model predicts the target location from the joint attention over the "
                             "input instruction and world state.")
    parser.add_argument("--no_cuda", dest="no_cuda", default=False, action="store_true",
                        help="Whether to use cuda if avaliable.")
    
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args
