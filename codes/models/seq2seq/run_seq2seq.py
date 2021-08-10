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

from decode_graphical_models import *
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
    max_decoding_steps, 
    pad_idx, 
    sos_idx,
    eos_idx, 
    max_examples_to_evaluate,
    device
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
    i = 0
    for step, batch in enumerate(data_iterator):
        
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        
        # derivation_spec
        # situation_spec
        input_sequence, target_sequence, situation,             agent_positions, target_positions,             input_lengths, target_lengths = batch
        
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
            
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model(
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
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        if model(tag="auxiliary_task"):
            pass
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


# In[ ]:


def evaluate(
    data_iterator,
    model, 
    max_decoding_steps, 
    pad_idx,
    sos_idx,
    eos_idx,
    max_examples_to_evaluate,
    device
):
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate, device=device):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))


# In[ ]:


def train(
    data_path: str, 
    data_directory: str, 
    generate_vocabularies: bool, 
    input_vocab_path: str,   
    target_vocab_path: str, 
    embedding_dimension: int, 
    num_encoder_layers: int, 
    encoder_dropout_p: float,
    encoder_bidirectional: bool, 
    training_batch_size: int, 
    test_batch_size: int, 
    max_decoding_steps: int,
    num_decoder_layers: int, 
    decoder_dropout_p: float, 
    cnn_kernel_size: int, 
    cnn_dropout_p: float,
    cnn_hidden_num_channels: int, 
    simple_situation_representation: bool, 
    decoder_hidden_size: int,
    encoder_hidden_size: int, 
    learning_rate: float, 
    adam_beta_1: float, 
    adam_beta_2: float, 
    lr_decay: float,
    lr_decay_steps: int, 
    resume_from_file: str, 
    max_training_iterations: int, 
    output_directory: str,
    print_every: int, 
    evaluate_every: int, 
    conditional_attention: bool, 
    auxiliary_task: bool,
    weight_target_loss: float, 
    attention_type: str, 
    k: int, 
    max_training_examples=None, 
    seed=42, **kwargs
):
    cfg = locals().copy()

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    from pathlib import Path
    # the output directory name is generated on-the-fly.
    run_name = f"seq2seq_seed_{seed}_max_itr_{max_training_iterations}"
    output_directory = os.path.join(output_directory, run_name)
    logger.info(f"Create the output directory if not exist: {output_directory}")
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading all data into memory...")
    logger.info(f"Reading dataset from file: {data_path}...")
    data_json = json.load(open(data_path, "r"))
    
    logger.info("Loading Training set...")
    training_set = ReaSCANDataset(
        data_json, data_directory, split="train",
        input_vocabulary_file=input_vocab_path,
        target_vocabulary_file=target_vocab_path,
        generate_vocabulary=generate_vocabularies, k=k
    )
    training_set.read_dataset(
        max_examples=max_training_examples,
        simple_situation_representation=simple_situation_representation
    )
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = ReaSCANDataset(
        data_json, data_directory, split="dev",
        input_vocabulary_file=input_vocab_path,
        target_vocabulary_file=target_vocab_path,
        generate_vocabulary=generate_vocabularies, k=0
    )
    test_set.read_dataset(
        max_examples=None,
        simple_situation_representation=simple_situation_representation
    )

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    
    # create modell based on our dataset.
    model = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size,
                  num_cnn_channels=training_set.image_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  **cfg)
    
    # gpu setups
    use_cuda = True if torch.cuda.is_available() and not isnotebook() else False
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device}, and we recognize {n_gpu} gpu(s) in total.")

    # optimizer
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))
    
    
    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = -99
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))
    
    # Loading dataset and preprocessing a bit.
    train_data, _ = training_set.get_dataset()
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.training_batch_size)
    test_data, _ = test_set.get_dataset()
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    
    if use_cuda and n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    # graphical model
    train_max_decoding_steps = int(training_set.get_max_seq_length_target())
    logger.info(f"==== WARNING ====")
    logger.info(f"MAX_DECODING_STEPS for Training: {train_max_decoding_steps}")
    logger.info(f"==== WARNING ====")
    g_model = ReaSCANMultiModalLSTMCompGraph(
         model,
         train_max_decoding_steps,
         cache_results=False
    )
    
    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        for step, batch in enumerate(train_dataloader):
            input_batch, target_batch, situation_batch,                 agent_positions_batch, target_positions_batch,                 input_lengths_batch, target_lengths_batch = batch
            is_best = False
            model.train()
            
            input_max_seq_lens = max(input_lengths_batch)[0]
            target_max_seq_lens = max(target_lengths_batch)[0]
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            situation_batch = situation_batch.to(device)
            agent_positions_batch = agent_positions_batch.to(device)
            target_positions_batch = target_positions_batch.to(device)
            input_lengths_batch = input_lengths_batch.to(device)
            target_lengths_batch = target_lengths_batch.to(device)
            
            # Instead of calling forward(), we call the graph model wrapper.
            input_dict = {
                "commands_input": input_batch, 
                "commands_lengths": input_lengths_batch,
                "situations_input": situation_batch,
                "target_batch": target_batch,
                "target_lengths": target_lengths_batch,
            }
            g_input_dict = GraphInput(input_dict, batched=True, batch_dim=0, cache_results=False)
            target_scores = g_model.compute(g_input_dict)
            loss = model(
                loss_target_scores=target_scores, 
                loss_target_batch=target_batch,
                tag="loss"
            )
            if use_cuda and n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            # we need to average over actual length to get rid of padding losses.
            # loss /= target_lengths_batch.sum()
            if auxiliary_task:
                target_loss = 0
                pass
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss
            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model(
                is_best=is_best,
                tag="update_state"
            )
            
            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model(
                    loss_target_scores=target_scores, 
                    loss_target_batch=target_batch,
                    tag="get_metrics"
                )
                if auxiliary_task:
                    pass
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")
                    accuracy, exact_match, target_accuracy = evaluate(
                        test_dataloader, model=model,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"],
                        device=device
                    )
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model(
                            accuracy=accuracy, exact_match=exact_match, 
                            is_best=is_best,
                            tag="update_state"
                        )
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        pass
                        # model.save_checkpoint(file_name=file_name, is_best=is_best,
                        #                       optimizer_state_dict=optimizer.state_dict())
                
            training_iteration += 1
            if training_iteration > max_training_iterations:
                break


# In[ ]:


def main(flags):
    
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
        train(data_path=data_path, **flags)  


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

