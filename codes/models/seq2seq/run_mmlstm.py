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
        
        input_sequence, target_sequence, situation,             agent_positions, target_positions,             input_lengths, target_lengths,             dual_input_sequence, dual_target_sequence, dual_situation,             dual_agent_positions, dual_target_positions,             dual_input_lengths, dual_target_lengths = batch
        
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
        yield (input_sequence, output_sequence, target_sequence, auxiliary_accuracy_target)

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
    for input_sequence, output_sequence, target_sequence, aux_acc_target in predict(
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
    args,
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
    # counterfactual training arguments
    run_name: str,
    cf_mode: str,
    cf_sample_p: float,
    checkpoint_save_every: int,
    include_cf_loss: bool,
    include_task_loss: bool,
    cf_loss_weight: float,
    is_wandb: bool,
    intervene_attribute: int,
    intervene_time: int,
    intervene_dimension_size: int,
    include_cf_auxiliary_loss: bool,
    intervene_method: str,
    no_cuda: bool,
    restrict_sampling: str,
    max_training_examples=None, 
    seed=42,
    **kwargs
):
    # we at least need to have one kind of loss.
    logger.info(f"LOSS CONFIG: include_task_loss={include_task_loss}, "
                f"include_cf_loss={include_cf_loss} with cf_loss_weight = {cf_loss_weight}...")
    
    cfg = locals().copy()

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_gpu = torch.cuda.device_count()
    
    from pathlib import Path
    # the output directory name is generated on-the-fly.
    dataset_name = data_directory.strip("/").split("/")[-1]
    run_name = f"counterfactual_{dataset_name}_seed_{seed}_lr_{learning_rate}_attr_{intervene_attribute}_size_{intervene_dimension_size}_cf_loss_{include_cf_loss}_aux_loss_{include_cf_auxiliary_loss}_restrict_{restrict_sampling}"
    output_directory = os.path.join(output_directory, run_name)
    cfg["output_directory"] = output_directory
    logger.info(f"Create the output directory if not exist: {output_directory}")
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # initialize w&b in the beginning.
    if is_wandb:
        logger.warning("Enabling wandb for tensorboard logging...")
        import wandb
        run = wandb.init(
            project="ReaSCAN-Causal", 
            entity="wuzhengx",
            name=run_name,
        )
        wandb.config.update(args)
    
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
    
    # some important variables.
    grid_size = training_set.grid_size
    target_position_size = 2*grid_size - 1
    
    # create modell based on our dataset.
    model = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size,
                  num_cnn_channels=training_set.image_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  target_position_size=target_position_size,
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
    train_data, _ = training_set.get_dual_dataset()
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.training_batch_size)
    test_data, _ = test_set.get_dual_dataset()
    test_dataloader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    
    if use_cuda and n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    # graphical model
    train_max_decoding_steps = int(training_set.get_max_seq_length_target())
    logger.info(f"==== WARNING ====")
    logger.info(f"MAX_DECODING_STEPS for Training: {train_max_decoding_steps}")
    logger.info(f"==== WARNING ====")

    hi_model = HighLevelModel(
        # None
    )
    hi_model.to(device)
    logger.info("Finish loading both low and high models..")
    
    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        for step, batch in enumerate(train_dataloader):
            # main batch
            input_batch, target_batch, situation_batch,                 agent_positions_batch, target_positions_batch,                 input_lengths_batch, target_lengths_batch,                 dual_input_batch, dual_target_batch, dual_situation_batch,                 dual_agent_positions_batch, dual_target_positions_batch,                 dual_input_lengths_batch, dual_target_lengths_batch = batch
            is_best = False
            model.train()
            
            is_best = False
            model.train()
            
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            situation_batch = situation_batch.to(device)
            agent_positions_batch = agent_positions_batch.to(device)
            target_positions_batch = target_positions_batch.to(device)
            input_lengths_batch = input_lengths_batch.to(device)
            target_lengths_batch = target_lengths_batch.to(device)
            
            dual_input_max_seq_lens = max(dual_input_lengths_batch)[0]
            dual_target_max_seq_lens = max(dual_target_lengths_batch)[0]
            
            # let us allow shuffling here, so that we have more diversity.
            perm_idx = torch.randperm(dual_input_batch.size()[0])
            dual_input_batch = dual_input_batch.index_select(dim=0, index=perm_idx)
            dual_target_batch = dual_target_batch.index_select(dim=0, index=perm_idx)
            dual_situation_batch = dual_situation_batch.index_select(dim=0, index=perm_idx)
            dual_agent_positions_batch = dual_agent_positions_batch.index_select(dim=0, index=perm_idx)
            dual_target_positions_batch = dual_target_positions_batch.index_select(dim=0, index=perm_idx)
            dual_input_lengths_batch = dual_input_lengths_batch.index_select(dim=0, index=perm_idx)
            dual_target_lengths_batch = dual_target_lengths_batch.index_select(dim=0, index=perm_idx)
            dual_input_batch = dual_input_batch.to(device)
            dual_target_batch = dual_target_batch.to(device)
            dual_situation_batch = dual_situation_batch.to(device)
            dual_agent_positions_batch = dual_agent_positions_batch.to(device)
            dual_target_positions_batch = dual_target_positions_batch.to(device)
            dual_input_lengths_batch = dual_input_lengths_batch.to(device)
            dual_target_lengths_batch = dual_target_lengths_batch.to(device)
            
            loss = None
            task_loss = None
            cf_loss = None
            cf_position_loss = None
            
            # we use the main hidden to track.
            task_encoded_image = model(
                situations_input=situation_batch,
                tag="situation_encode"
            )
            task_hidden, task_encoder_outputs = model(
                commands_input=input_batch, 
                commands_lengths=input_lengths_batch,
                tag="command_input_encode_no_dict"
            )
            task_hidden = model(
                command_hidden=task_hidden,
                tag="initialize_hidden"
            )
            task_projected_keys_visual = model(
                encoded_situations=task_encoded_image,
                tag="projected_keys_visual"
            )
            task_projected_keys_textual = model(
                command_encoder_outputs=task_encoder_outputs["encoder_outputs"],
                tag="projected_keys_textual"
            )
            task_outputs = []
            for j in range(train_max_decoding_steps):
                task_token = target_batch[:,j]
                (task_output, task_hidden) = model(
                    lstm_input_tokens_sorted=task_token,
                    lstm_hidden=task_hidden,
                    lstm_projected_keys_textual=task_projected_keys_textual,
                    lstm_commands_lengths=input_lengths_batch,
                    lstm_projected_keys_visual=task_projected_keys_visual,
                    tag="_lstm_step_fxn"
                )
                task_output = F.log_softmax(task_output, dim=-1)
                task_outputs += [task_output]
            target_scores = torch.stack(task_outputs, dim=1)
            task_loss = model(
                loss_target_scores=target_scores, 
                loss_target_batch=target_batch,
                tag="loss"
            )
            if use_cuda and n_gpu > 1:
                task_loss = task_loss.mean() # mean() to average on multi-gpu.
            
            input_max_seq_lens = max(input_lengths_batch)[0]
            target_max_seq_lens = max(target_lengths_batch)[0]
            dual_target_max_seq_lens = max(dual_target_lengths_batch)[0]
            intervene_attribute = random.choice([0,1])
            intervene_time = 1
            intervene_with_time = intervene_time
            
            batch_size = agent_positions_batch.size(0)
            intervened_target_batch = []
            intervened_target_lengths_batch = []

            high_hidden_states = hi_model(
                agent_positions_batch=agent_positions_batch.unsqueeze(dim=-1), 
                target_positions_batch=target_positions_batch.unsqueeze(dim=-1), 
                tag="situation_encode"
            )
            high_actions = torch.zeros(
                high_hidden_states.size(0), 1
            ).long().to(device)
            dual_high_hidden_states = hi_model(
                agent_positions_batch=dual_agent_positions_batch.unsqueeze(dim=-1), 
                target_positions_batch=dual_target_positions_batch.unsqueeze(dim=-1), 
                tag="situation_encode"
            )
            dual_high_actions = torch.zeros(
                dual_high_hidden_states.size(0), 1
            ).long().to(device)
            # get the intercepted dual hidden states.
            for j in range(intervene_with_time):
                dual_high_hidden_states, dual_high_actions = hi_model(
                    hmm_states=dual_high_hidden_states, 
                    hmm_actions=dual_high_actions, 
                    tag="_hmm_step_fxn"
                )
            # main intervene for loop.
            cf_high_hidden_states = high_hidden_states
            cf_high_actions = high_actions
            intervened_target_batch = [torch.ones(high_hidden_states.size(0), 1).long().to(device)] # SOS tokens
            intervened_target_lengths_batch = torch.zeros(high_hidden_states.size(0), 1).long().to(device)
            # we need to take of the SOS and EOS tokens.
            for j in range(train_max_decoding_steps-1):
                # intercept like antra!
                if j == intervene_time-1:
                    # we need idle once by getting the states but not continue the HMM!
                    cf_high_hidden_states, _ = hi_model(
                        hmm_states=cf_high_hidden_states, 
                        hmm_actions=cf_high_actions, 
                        tag="_hmm_step_fxn"
                    )
                    # we also include a probe loss.
                    if include_cf_auxiliary_loss:
                        true_target_positions = dual_high_hidden_states+5

                    # only swap out this part.
                    cf_high_hidden_states[:,intervene_attribute] = dual_high_hidden_states[:,intervene_attribute]
                    cf_high_actions = torch.zeros(
                        dual_high_hidden_states.size(0), 1
                    ).long().to(device)
                    # comment out two lines below if it is not for testing.
                    # cf_high_hidden_states = dual_high_hidden_states
                    # cf_high_actions = dual_high_actions
                cf_high_hidden_states, cf_high_actions = hi_model(
                    hmm_states=cf_high_hidden_states, 
                    hmm_actions=cf_high_actions, 
                    tag="_hmm_step_fxn"
                )
                # record the output for loss calculation.
                intervened_target_batch += [cf_high_actions]
                intervened_target_lengths_batch += (cf_high_actions!=0).long()
            intervened_target_lengths_batch += 2
            # we do not to increase the EOS for non-ending sequences
            for i in range(high_hidden_states.size(0)):
                if intervened_target_lengths_batch[i,0] == train_max_decoding_steps+1:
                    intervened_target_lengths_batch[i,0] = train_max_decoding_steps
            intervened_target_batch = torch.cat(intervened_target_batch, dim=-1)
            for i in range(high_hidden_states.size(0)):
                if intervened_target_batch[i,intervened_target_lengths_batch[i,0]-1] == 0:
                    intervened_target_batch[i,intervened_target_lengths_batch[i,0]-1] = 2
            # intervened data.
            intervened_target_batch = intervened_target_batch.clone()
            intervened_target_lengths_batch = intervened_target_lengths_batch.clone()
            # correct the length.
            intervened_target_lengths_batch[intervened_target_lengths_batch>train_max_decoding_steps] = train_max_decoding_steps
            
            # Major update:
            # We need to batch these operations.
            intervened_scores_batch = []
            # here, we only care where those intervened target is different
            # from the original main target.
            idx_generator = []
            for i in range(batch_size):
                match_target_intervened = intervened_target_batch[i,:intervened_target_lengths_batch[i,0]]
                match_target_main = target_batch[i,:target_lengths_batch[i,0]]
                # is_bad_intervened = torch.equal(match_target_intervened, match_target_main)
                is_bad_intervened = (intervene_time>(target_lengths_batch[i,0]-2)) or                     (intervene_with_time>(dual_target_lengths_batch[i,0]-2))
                if is_bad_intervened:
                    continue # we need to skip these.
                else:
                    # we also need to make sure no testing time samples have been encountered before.
                    if restrict_sampling == "by_direction":
                        row_diff = saved_intervened_high_hidden_states[i][0]
                        col_diff = saved_intervened_high_hidden_states[i][1]
                        if row_diff > 0 and col_diff < 0:
                            # skip these examples in new direction split.
                            continue
                        else:
                            idx_generator += [i]
                    elif restrict_sampling == "by_length":
                        if intervened_target_lengths_batch[i,0] >= 12:
                            # skip these examples in new action length split.
                            continue
                        else:
                            idx_generator += [i]
                    elif restrict_sampling == "none":
                        idx_generator += [i]
                    else:
                        assert False

            # Let us get rid of antra, using a very simple for loop
            # to do this intervention.
            idx_selected = []
            if len(idx_generator) > 0:
                # overwrite a bit.
                cf_sample_per_batch = min(cf_sample_per_batch, len(idx_generator))
                idx_selected = random.sample(idx_generator, k=cf_sample_per_batch)
                intervened_target_batch_selected = []

                # filter based on selection all together!
                situation_batch = situation_batch[idx_selected]
                input_batch = input_batch[idx_selected]
                input_lengths_batch = input_lengths_batch[idx_selected]
                dual_situation_batch = dual_situation_batch[idx_selected]
                dual_input_batch = dual_input_batch[idx_selected]
                dual_input_lengths_batch = dual_input_lengths_batch[idx_selected]
                dual_target_batch = dual_target_batch[idx_selected]
                intervened_target_lengths_batch = intervened_target_lengths_batch[idx_selected]
                intervened_target_batch = intervened_target_batch[idx_selected]
                if include_cf_auxiliary_loss:
                    true_target_positions = true_target_positions[idx_selected]
                
                # we use the main hidden to track.
                encoded_image = model(
                    situations_input=situation_batch,
                    tag="situation_encode"
                )
                main_hidden, encoder_outputs = model(
                    commands_input=input_batch, 
                    commands_lengths=input_lengths_batch,
                    tag="command_input_encode_no_dict"
                )

                main_hidden = model(
                    command_hidden=main_hidden,
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

                # dual setup.
                dual_input_batch = dual_input_batch[:,:dual_input_max_seq_lens]
                dual_target_batch = dual_target_batch[:,:dual_target_max_seq_lens]
                dual_encoded_image = model(
                    situations_input=dual_situation_batch,
                    tag="situation_encode"
                )
                dual_hidden, dual_encoder_outputs = model(
                    commands_input=dual_input_batch, 
                    commands_lengths=dual_input_lengths_batch,
                    tag="command_input_encode_no_dict"
                )

                dual_hidden = model(
                    command_hidden=dual_hidden,
                    tag="initialize_hidden"
                )
                dual_projected_keys_visual = model(
                    encoded_situations=dual_encoded_image,
                    tag="projected_keys_visual"
                )
                dual_projected_keys_textual = model(
                    command_encoder_outputs=dual_encoder_outputs["encoder_outputs"],
                    tag="projected_keys_textual"
                )

                # get the intercepted dual hidden states.
                for j in range(intervene_with_time):
                    (dual_output, dual_hidden) = model(
                        lstm_input_tokens_sorted=dual_target_batch[:,j],
                        lstm_hidden=dual_hidden,
                        lstm_projected_keys_textual=dual_projected_keys_textual,
                        lstm_commands_lengths=dual_input_lengths_batch,
                        lstm_projected_keys_visual=dual_projected_keys_visual,
                        tag="_lstm_step_fxn"
                    )
                
                # main intervene for loop.
                cf_hidden = main_hidden
                cf_outputs = []
                for j in range(intervened_target_batch.shape[1]):
                    cf_token=intervened_target_batch[:,j]
                    # intercept like antra!
                    if j == intervene_time-1:
                        # we need idle once by getting the states but not continue the HMM!
                        (_, cf_hidden) = model(
                            lstm_input_tokens_sorted=cf_token,
                            lstm_hidden=cf_hidden,
                            lstm_projected_keys_textual=projected_keys_textual,
                            lstm_commands_lengths=input_lengths_batch,
                            lstm_projected_keys_visual=projected_keys_visual,
                            tag="_lstm_step_fxn"
                        )
                        # we also include a probe loss.
                        if include_cf_auxiliary_loss:
                            x_s_idx = 0
                            y_s_idx = intervene_dimension_size
                            y_e_idx = 2*intervene_dimension_size
                            cf_target_positions_x = model(
                                position_hidden=dual_hidden[0][:,:,x_s_idx:y_s_idx].squeeze(dim=1),
                                cf_auxiliary_task_tag="x",
                                tag="cf_auxiliary_task"
                            )
                            cf_target_positions_x = F.log_softmax(cf_target_positions_x, dim=-1)
                            cf_target_positions_y = model(
                                position_hidden=dual_hidden[0][:,:,y_s_idx:y_e_idx].squeeze(dim=1),
                                cf_auxiliary_task_tag="x",
                                tag="cf_auxiliary_task"
                            )
                            cf_target_positions_y = F.log_softmax(cf_target_positions_y, dim=-1)
                            loss_position_x = model(
                                loss_pred_target_positions=cf_target_positions_x,
                                loss_true_target_positions=true_target_positions[:,0],
                                tag="cf_auxiliary_task_loss"
                            )
                            loss_position_y = model(
                                loss_pred_target_positions=cf_target_positions_y,
                                loss_true_target_positions=true_target_positions[:,1],
                                tag="cf_auxiliary_task_loss"
                            )
                            cf_position_loss = loss_position_x + loss_position_y
                            if use_cuda and n_gpu > 1:
                                cf_position_loss = cf_position_loss.mean() # mean() to average on multi-gpu.
                            # some metrics
                            metrics_position_x = model(
                                loss_pred_target_positions=cf_target_positions_x,
                                loss_true_target_positions=true_target_positions[:,0],
                                tag="cf_auxiliary_task_metrics"
                            )
                            metrics_position_y = model(
                                loss_pred_target_positions=cf_target_positions_y,
                                loss_true_target_positions=true_target_positions[:,1],
                                tag="cf_auxiliary_task_metrics"
                            )
                            
                        # intervene!
                        s_idx = intervene_attribute*intervene_dimension_size
                        e_idx = (intervene_attribute+1)*intervene_dimension_size
                        if intervene_method == "cat":
                            updated_hidden = torch.cat(
                                [
                                   cf_hidden[0][:,:,:s_idx],
                                   dual_hidden[0][:,:,s_idx:e_idx],
                                   cf_hidden[0][:,:,e_idx:]
                                ], dim=-1
                            )
                            cf_hidden = (updated_hidden, cf_hidden[1])
                        elif intervene_method == "inplace":
                            cf_hidden[0][:,:,s_idx:e_idx] = dual_hidden[0][:,:,s_idx:e_idx] # only swap out this part.
                        else:
                            assert False
                        # WARNING: this is a special way to bypassing the state
                        # updates during intervention!
                        cf_token = None
                    (cf_output, cf_hidden) = model(
                        lstm_input_tokens_sorted=cf_token,
                        lstm_hidden=cf_hidden,
                        lstm_projected_keys_textual=projected_keys_textual,
                        lstm_commands_lengths=input_lengths_batch,
                        lstm_projected_keys_visual=projected_keys_visual,
                        tag="_lstm_step_fxn"
                    )
                    # record the output for loss calculation.
                    cf_output = cf_output.unsqueeze(0)
                    cf_outputs += [cf_output]
                cf_outputs = torch.cat(cf_outputs, dim=0)
                intervened_scores_batch = cf_outputs.transpose(0, 1) # [batch_size, max_target_seq_length, target_vocabulary_size]
                
                # Counterfactual loss
                intervened_scores_batch = F.log_softmax(intervened_scores_batch, dim=-1)
                cf_loss = model(
                    loss_target_scores=intervened_scores_batch, 
                    loss_target_batch=intervened_target_batch,
                    tag="loss"
                )
                if use_cuda and n_gpu > 1:
                    cf_loss = cf_loss.mean() # mean() to average on multi-gpu.
            
            loss = task_loss
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
                if is_wandb:
                    wandb.log({'train/training_iteration': training_iteration})
                    wandb.log({'train/task_loss': loss})
                    wandb.log({'train/task_accuracy': accuracy})
                    wandb.log({'train/task_exact_match': exact_match})
                    
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
                    if is_wandb:
                        wandb.log({'eval/accuracy': accuracy})
                        wandb.log({'eval/exact_match': exact_match})
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model(
                            accuracy=accuracy, exact_match=exact_match, 
                            is_best=is_best,
                            tag="update_state"
                        )
                    file_name = f"checkpoint-{training_iteration}.pth.tar"
                    model.save_checkpoint(file_name=file_name, is_best=is_best,
                                          optimizer_state_dict=optimizer.state_dict())
                
            training_iteration += 1
            if training_iteration > max_training_iterations:
                break


# In[ ]:


def main(flags, args):
    
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
        train(data_path=data_path, args=args, **flags)  


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
    main(input_flags, args)

