#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import logging
import os
import torch
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import time
import random
import torch.nn.functional as F
from tqdm import tqdm, trange

from seq2seq.model import *
from decode_abstract_models import *
from seq2seq.ReaSCAN_dataset import *
from seq2seq.helpers import *
from torch.optim.lr_scheduler import LambdaLR

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# In[ ]:


def predict(
    data_iterator, 
    model, 
    hi_model,
    max_decoding_steps, 
    pad_idx, 
    sos_idx,
    eos_idx, 
    max_examples_to_evaluate,
    device,
    intervene_time,
    intervene_dimension_size,
):
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for step, batch in enumerate(tqdm(data_iterator)):
        
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        
        # derivation_spec
        # situation_spec
        input_sequence, target_sequence, situation,         agent_positions, target_positions,         input_lengths, target_lengths,         dual_input_sequence, dual_target_sequence, dual_situation,         dual_agent_positions, dual_target_positions,         dual_input_lengths, dual_target_lengths,         intervened_main_swap_index, intervened_dual_swap_index,         intervened_main_shape_index,intervened_dual_shape_index,         intervened_target_batch, intervened_swap_attr,         intervened_target_lengths_batch, size_class, color_class, shape_class = batch
        
        input_max_seq_lens = max(input_lengths)[0]
        target_max_seq_lens = max(target_lengths)[0]
        
        input_sequence = input_sequence.to(device)
        target_sequence = target_sequence.to(device)
        situation = situation.to(device)
        agent_positions = agent_positions.to(device)
        target_positions = target_positions.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        # We need to chunk
        input_sequence = input_sequence[:,:input_max_seq_lens]
        target_sequence = target_sequence[:,:target_max_seq_lens]
        
        # in the evaluation phase, i think we can actually
        # use the model itself not the graphical model.
        # ENCODE
        encoded_image = model(
            situations_input=situation,
            tag="situation_encode"
        )
        hidden, encoder_outputs = model(
            commands_input=input_sequence, 
            commands_lengths=input_lengths,
            tag="command_input_encode_no_dict"
        )

        # DECODER INIT
        hidden = model(
            command_hidden=hidden,
            tag="initialize_hidden"
        )
        projected_keys_visual = model(
            encoded_situations=encoded_image,
            tag="projected_keys_visual"
        )
        projected_keys_textual = model(
            command_encoder_outputs=encoder_outputs["encoder_outputs"],
            tag="projected_keys_textual"
        )
        
        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            
            (output, hidden) = model(
                lstm_input_tokens_sorted=token,
                lstm_hidden=hidden,
                lstm_projected_keys_textual=projected_keys_textual,
                lstm_commands_lengths=input_lengths,
                lstm_projected_keys_visual=projected_keys_visual,
                tag="_lstm_step_fxn"
            )
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]

            output_sequence.append(token.data[0].item())
            
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()

        auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0

        yield (
            input_sequence, output_sequence, target_sequence, auxiliary_accuracy_target
        )

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


# In[5]:


def counterfactual_predict(
    data_iterator, 
    model, hi_model,
    max_decoding_steps, 
    eval_max_decoding_steps,
    pad_idx, 
    sos_idx,
    eos_idx, 
    max_examples_to_evaluate,
    device,
    intervene_attribute=-1,
    intervene_time=-1,
    intervene_dimension_size=25,
    intervene_position="last_hidden",
) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    total_count = 0
    bad_count = 0
    
    random_attr = False
    random_time = False
    if intervene_attribute == -1:
        random_attr = True
    if intervene_time == -1:
        random_time = True
    for step, batch in enumerate(tqdm(data_iterator)):
        
        total_count += 1
        if max_examples_to_evaluate:
            if total_count > max_examples_to_evaluate:
                break

        # main batch
        input_batch, target_batch, situation_batch,             agent_positions_batch, target_positions_batch,             input_lengths_batch, target_lengths_batch,             dual_input_batch, dual_target_batch, dual_situation_batch,             dual_agent_positions_batch, dual_target_positions_batch,             dual_input_lengths_batch, dual_target_lengths_batch,             intervened_main_swap_index, intervened_dual_swap_index,             intervened_main_shape_index,intervened_dual_shape_index,             intervened_target_batch, intervened_swap_attr,             intervened_target_lengths_batch, size_class, color_class, shape_class = batch

        
        assert input_batch.size(0) == 1
        
        input_max_seq_lens = max(input_lengths_batch)[0]
        target_max_seq_lens = max(target_lengths_batch)[0]
        
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        situation_batch = situation_batch.to(device)
        agent_positions_batch = agent_positions_batch.to(device)
        target_positions_batch = target_positions_batch.to(device)
        input_lengths_batch = input_lengths_batch.to(device)
        target_lengths_batch = target_lengths_batch.to(device)

        dual_input_max_seq_lens = max(dual_input_lengths_batch)[0]
        dual_target_max_seq_lens = max(dual_target_lengths_batch)[0]
        dual_input_batch = dual_input_batch.to(device)
        dual_target_batch = dual_target_batch.to(device)
        dual_situation_batch = dual_situation_batch.to(device)
        dual_agent_positions_batch = dual_agent_positions_batch.to(device)
        dual_target_positions_batch = dual_target_positions_batch.to(device)
        dual_input_lengths_batch = dual_input_lengths_batch.to(device)
        dual_target_lengths_batch = dual_target_lengths_batch.to(device)
        # intervention data for novel attribute.
        intervened_main_swap_index = intervened_main_swap_index.to(device)
        intervened_dual_swap_index = intervened_dual_swap_index.to(device)
        intervened_main_shape_index = intervened_dual_swap_index.to(device)
        intervened_dual_shape_index = intervened_dual_shape_index.to(device)
        intervened_target_batch = intervened_target_batch.to(device)
        intervened_swap_attr = intervened_swap_attr.to(device)
        intervened_target_lengths_batch = intervened_target_lengths_batch.to(device)
        size_class = size_class.to(device)
        color_class = color_class.to(device)
        shape_class = shape_class.to(device)
            
        # Step 1: using a for loop to get the hidden states from the dual data
        input_batch = input_batch[:,:input_max_seq_lens]
        target_batch = target_batch[:,:target_max_seq_lens]
        dual_target_batch = dual_target_batch[:,:dual_target_max_seq_lens]
        
        # what to intervene
        """
        For the sake of quick training, for a single batch,
        we select the same attribute to intervenen on:
        0: x
        1: y
        2: orientation
        """
        input_max_seq_lens = max(input_lengths_batch)[0]
        target_max_seq_lens = max(target_lengths_batch)[0]
        dual_target_max_seq_lens = max(dual_target_lengths_batch)[0]
        min_len = min(target_max_seq_lens, dual_target_max_seq_lens)
        
        batch_size = input_batch.size(0)
        # calculate cf loss.
        # just like other cf, we need to first filter out some examples.
        idx_selected = []
        for i in range(batch_size):
            if intervened_main_swap_index[i] != -1:
                idx_selected += [i]
            else:
                assert intervened_target_lengths_batch[i] == 0 # this is a validation.
        if len(idx_selected) > 0:
            situation_batch = situation_batch[idx_selected]
            input_batch = input_batch[idx_selected]
            input_lengths_batch = input_lengths_batch[idx_selected]
            dual_situation_batch = dual_situation_batch[idx_selected]
            dual_input_batch = dual_input_batch[idx_selected]
            dual_input_lengths_batch = dual_input_lengths_batch[idx_selected]
            dual_target_batch = dual_target_batch[idx_selected]
            intervened_target_lengths_batch = intervened_target_lengths_batch[idx_selected]
            intervened_target_batch = intervened_target_batch[idx_selected]
            intervened_main_swap_index = intervened_main_swap_index[idx_selected]
            intervened_dual_swap_index = intervened_dual_swap_index[idx_selected]
            intervened_main_shape_index = intervened_main_shape_index[idx_selected]
            intervened_dual_shape_index = intervened_dual_shape_index[idx_selected]
            intervened_swap_attr = intervened_swap_attr[idx_selected]
        
            if intervene_position == "embedding":
                assert False # not supporting it for this version.
            elif intervene_position == "hidden":
                assert False # not supporting it for this version.
            elif intervene_position == "last_hidden":
                commands_embedding = model(
                    commands_input=input_batch, 
                    tag="command_input_encode_embedding"
                )
                dual_commands_embedding = model(
                    commands_input=dual_input_batch, 
                    tag="command_input_encode_embedding"
                )
                hidden, encoder_outputs = model(
                    commands_embedding=commands_embedding, 
                    commands_lengths=input_lengths_batch,
                    tag="command_input_encode_no_dict_with_embedding"
                )
                dual_hidden, dual_encoder_outputs = model(
                    commands_embedding=dual_commands_embedding, 
                    commands_lengths=input_lengths_batch,
                    tag="command_input_encode_no_dict_with_embedding"
                )
                # intervene on lstm hidden.
                intervened_encoder_outputs = encoder_outputs["encoder_outputs"]
                # intervene on init hidden as well.
                intervened_hidden = hidden
                for i in range(len(idx_selected)):
                    assert intervened_main_swap_index[i] != -1
                    start_idx = intervened_swap_attr[i]*intervene_dimension_size
                    end_idx = (intervened_swap_attr[i]+1)*intervene_dimension_size
                    intervened_encoder_outputs[
                        i,intervened_main_shape_index[i]:intervened_main_shape_index[i]+1,start_idx:end_idx
                    ] = dual_encoder_outputs["encoder_outputs"][
                        i,intervened_dual_shape_index[i]:intervened_dual_shape_index[i]+1,start_idx:end_idx
                    ]
                    intervened_hidden[i,start_idx:end_idx] = hidden[i,start_idx:end_idx]
                hidden = model(
                    command_hidden=intervened_hidden,
                    tag="initialize_hidden"
                )
                encoded_image = model(
                    situations_input=situation_batch,
                    tag="situation_encode"
                )
                projected_keys_visual = model(
                    encoded_situations=encoded_image,
                    tag="projected_keys_visual"
                )
                projected_keys_textual = model(
                    command_encoder_outputs=intervened_encoder_outputs,
                    tag="projected_keys_textual"
                )

            # decoder which we do not touch at all!
            output_sequence = []
            cf_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
            cf_hidden = hidden
            for j in range(intervened_target_batch.shape[1]):
                (cf_output, cf_hidden) = model(
                    lstm_input_tokens_sorted=cf_token,
                    lstm_hidden=cf_hidden,
                    lstm_projected_keys_textual=projected_keys_textual,
                    lstm_commands_lengths=input_lengths_batch,
                    lstm_projected_keys_visual=projected_keys_visual,
                    tag="_lstm_step_fxn"
                )
                # record the output for loss calculation.
                cf_output = F.log_softmax(cf_output, dim=-1)
                cf_token = cf_output.max(dim=-1)[1]
                output_sequence.append(cf_token.data[0].item())
                # we need to stop.
                if cf_token == eos_idx:
                    break

            if output_sequence[-1] == eos_idx:
                output_sequence.pop()
            yield input_batch, dual_input_batch,                 output_sequence, intervened_target_batch[:,:intervened_target_lengths_batch[0]], target_batch, dual_target_batch, 0.0
        


# In[5]:


def array_to_sentence(sentence_array, vocab):
    return [vocab.itos[word_idx] for word_idx in sentence_array]

def predict_and_save(
    dataset: ReaSCANDataset, 
    model: nn.Module, 
    hi_model,
    output_file_path: str, 
    max_decoding_steps: int,
    device,
    max_testing_examples=None, 
    **kwargs
):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    # read-in datasets
    if cfg["kwargs"]["restrict_sampling"] == "none":
        restrict_sampling = None
    else:
        restrict_sampling = cfg["kwargs"]["restrict_sampling"]
    test_data, _ = dataset.get_dual_dataset(novel_attribute=True, restrict_sampling=restrict_sampling)
    data_iterator = DataLoader(test_data, batch_size=1, shuffle=False)
    eval_max_decoding_steps = max_decoding_steps
    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            
            if cfg["kwargs"]["task_evaluate"]:
                #################
                #
                # Task-based
                #
                #################
                exact_match_count = 0
                example_count = 0
                for input_sequence, output_sequence, target_sequence, aux_acc_target in predict(
                        data_iterator=data_iterator, model=model, hi_model=hi_model, 
                        max_decoding_steps=max_decoding_steps, 
                        pad_idx=dataset.target_vocabulary.pad_idx,
                        sos_idx=dataset.target_vocabulary.sos_idx, 
                        eos_idx=dataset.target_vocabulary.eos_idx, 
                        max_examples_to_evaluate=max_testing_examples, 
                        device=device,
                        intervene_time=cfg["kwargs"]["intervene_time"],
                        intervene_dimension_size=cfg["kwargs"]["intervene_dimension_size"],
                ):
                    example_count += 1
                    accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                    input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                    input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                    target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                    if accuracy == 100:
                        exact_match_count += 1
                    output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                                   "target": target_str_sequence,
                                   "accuracy": accuracy,
                                   "exact_match": True if accuracy == 100 else False,
                                  })      
                exact_match = (exact_match_count/example_count)*100.0
                logger.info(" Task Evaluation Exact Match: %5.2f " % (exact_match))
                if not cfg["kwargs"]["counterfactual_evaluate"]:
                    logger.info("Wrote predictions for {} examples.".format(example_count))
                    json.dump(output, outfile, indent=4)
                    return output_file_path
            else:
                logger.info("WARNING: Skipping regular task evaluation...")
            
            if cfg["kwargs"]["counterfactual_evaluate"]:
                #################
                #
                # Counterfactual
                #
                #################
                logger.info(" Starting our counterfactual analysis ... ")
                exact_match_count = 0
                example_count = 0
                for input_sequence, dual_input_sequence,                     output_sequence, target_sequence, original_target_sequence, original_dual_target_str_sequence,                     aux_acc_target in counterfactual_predict(
                        data_iterator=data_iterator, model=model, 
                        hi_model=hi_model,
                        max_decoding_steps=max_decoding_steps, 
                        eval_max_decoding_steps=eval_max_decoding_steps,
                        pad_idx=dataset.target_vocabulary.pad_idx,
                        sos_idx=dataset.target_vocabulary.sos_idx, 
                        eos_idx=dataset.target_vocabulary.eos_idx, 
                        max_examples_to_evaluate=max_testing_examples, 
                        device=device,
                        intervene_attribute=cfg["kwargs"]["intervene_attribute"],
                        intervene_time=cfg["kwargs"]["intervene_time"],
                        intervene_dimension_size=cfg["kwargs"]["intervene_dimension_size"],
                ):
                    example_count += 1
                    accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                    input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                    input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    dual_input_str_sequence = dataset.array_to_sentence(dual_input_sequence[0].tolist(), vocabulary="input")
                    dual_input_str_sequence = dual_input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>

                    target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                    original_target_str_sequence = dataset.array_to_sentence(original_target_sequence[0].tolist()[1:-1], vocabulary="target")
                    original_dual_target_str_sequence = dataset.array_to_sentence(original_dual_target_str_sequence[0].tolist()[1:-1], vocabulary="target")
                    target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                    if accuracy == 100:
                        exact_match_count += 1
                    output.append({"input": input_str_sequence, 
                                   "dual_input": dual_input_str_sequence,
                                   "prediction": output_str_sequence,
                                   "target": target_str_sequence,
                                   "main_target": original_target_str_sequence,
                                   "dual_target": original_dual_target_str_sequence,
                                   "accuracy": accuracy,
                                   "exact_match": True if accuracy == 100 else False,
                                   "position_accuracy": aux_acc_target})
                logger.info("Wrote predictions for {} examples including counterfactual ones.".format(example_count))
                json.dump(output, outfile, indent=4)
                exact_match = (exact_match_count/example_count)*100.0
                logger.info(" Counterfactual Evaluation Exact Match: %5.2f " % (exact_match))
        
    return output_file_path


# In[ ]:


def main(flags):
    
    random.seed(flags["seed"])
    torch.manual_seed(flags["seed"])
    np.random.seed(flags["seed"])
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))
    
    if not flags["simple_situation_representation"]:
        raise NotImplementedError("Full RGB input image not implemented. Implement or set "
                                  "--simple_situation_representation")
        
    # Some checks on the flags
    if not flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."
        
    if flags["test_batch_size"] > 1:
        raise NotImplementedError("Test batch size larger than 1 not implemented.")
        
    data_path = os.path.join(flags["data_directory"], "data-compositional-splits.txt")
    # quick check and fail fast!
    assert os.path.exists(data_path), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
        data_path)
    if flags["mode"] == "train":
        assert False # we don't allow train in the evaluation script.
    elif flags["mode"] == "test":
        logger.info("Loading all data into memory for evaluation...")
        logger.info(f"Reading dataset from file: {data_path}...")
        data_json = json.load(open(data_path, "r"))
    
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))
            
            test_set = ReaSCANDataset(
                data_json, flags["data_directory"], split=split,
                input_vocabulary_file=flags["input_vocab_path"],
                target_vocabulary_file=flags["target_vocab_path"],
                generate_vocabulary=False, k=flags["k"]
            )
            test_set.read_dataset(
                max_examples=flags["max_testing_examples"],
                simple_situation_representation=flags["simple_situation_representation"]
            )
            
            logger.info("Done Loading {} dataset split.".format(flags["split"]))
            logger.info("  Loaded {} examples.".format(test_set.num_examples))
            logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
            logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
            logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
            logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))
            
            grid_size = test_set.grid_size
            target_position_size = 2*grid_size - 1
            
            # create modell based on our dataset.
            model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
                          target_vocabulary_size=test_set.target_vocabulary_size,
                          num_cnn_channels=test_set.image_channels,
                          input_padding_idx=test_set.input_vocabulary.pad_idx,
                          target_pad_idx=test_set.target_vocabulary.pad_idx,
                          target_eos_idx=test_set.target_vocabulary.eos_idx,
                          target_position_size=target_position_size,
                          **flags)

            # gpu setups
            use_cuda = True if torch.cuda.is_available() and not isnotebook() and not flags["no_cuda"] else False
            device = torch.device("cuda" if use_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
            # logger.info(f"device: {device}, and we recognize {n_gpu} gpu(s) in total.")
            if use_cuda and n_gpu > 1:
                model = torch.nn.DataParallel(model)
            model.to(device)
            
            """
            We have two low model so that our computation is much faster.
            """
            eval_max_decoding_steps = flags["max_decoding_steps"] # we need to use this extended step to measure for eval.

            hi_model = HighLevelModel(
                # None
            )
            hi_model.to(device)
            # create high level model for counterfactual training.
            # hi_model = get_counter_compgraph(
            #     eval_max_decoding_steps,
            #     cache_results=False
            # )
            # logger.info("Finish loading both low and high models..")
            
            # optimizer
            # log_parameters(model)
            
            # Load model and vocabularies if resuming.
            evaluate_checkpoint = flags["evaluate_checkpoint"]
            if flags["evaluate_checkpoint"] == "":
                model_path = os.path.join(flags["resume_from_file"], "model_best.pth.tar")
            else:
                model_path = os.path.join(flags["resume_from_file"], f"checkpoint-{evaluate_checkpoint}.pth.tar")
            assert os.path.isfile(model_path), "No checkpoint found at {}".format(model_path)
            logger.info("Loading checkpoint from file at '{}'".format(model_path))
            model.load_model(device, model_path, strict=False)
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(model_path, start_iteration))
            output_file_name = "_".join([split, flags["output_file_name"]])
            output_file_path = os.path.join(flags["resume_from_file"], output_file_name)
            logger.info("All results will be saved to '{}'".format(output_file_path))
            
            output_file = predict_and_save(
                dataset=test_set, 
                model=model, hi_model=hi_model,
                output_file_path=output_file_path, 
                device=device,
                **flags
            )
            logger.info("Saved predictions to {}".format(output_file))
            


# In[ ]:


if __name__ == "__main__":
    
    # Loading arguments
    args = arg_parse()
    try:        
        get_ipython().run_line_magic('matplotlib', 'inline')
        is_jupyter = True
        args.max_training_examples = 10
        args.max_testing_examples = 1
        args.max_training_iterations = 5
        args.print_every = 1
        args.evaluate_every = 1
    except:
        is_jupyter = False
    
    input_flags = vars(args)
    main(flags=input_flags)

