import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import json
from model import *
from ReaSCAN_dataset import *
import torch.nn.functional as F
import torch
from antra.antra import *

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

"""
Below are the model for counterfactual training.
"""
    
def _generate_lstm_step_fxn_cf(step_module, i, max_decode_step, 
                            image_size=6, hidden_dim=100, 
                            vocab_size=6):
    """ 
    Generate a function for a layer in lstm.
    """

    def _lstm_step_fxn(hidden_states):
        if isnotebook():
            
            last_states = hidden_states
            batch_size = last_states.size(0)

            last_hidden = last_states[:,:hidden_dim].unsqueeze(dim=1).contiguous()
            last_cell = last_states[:,hidden_dim:hidden_dim*2].unsqueeze(dim=1).contiguous()
            input_tokens_sorted = last_states[:,hidden_dim*2:hidden_dim*2+max_decode_step].long().contiguous()
            
            commands_lengths = last_states[
                :,hidden_dim*2+max_decode_step:hidden_dim*2+max_decode_step+1
            ].long().contiguous()
            
            projected_keys_visual = last_states[
                :,hidden_dim*2+max_decode_step+1:hidden_dim*2+max_decode_step+1+image_size*image_size*hidden_dim
            ].reshape(
                batch_size, image_size*image_size, hidden_dim
            ).contiguous()
            
            _output = last_states[
                :,hidden_dim*2+max_decode_step+1+image_size*image_size*hidden_dim:hidden_dim*2+max_decode_step+1+image_size*image_size*hidden_dim+vocab_size
            ].contiguous()
            
            projected_keys_textual = last_states[
                :,hidden_dim*2+max_decode_step+1+image_size*image_size*hidden_dim+vocab_size:
            ].reshape(
                batch_size, -1, hidden_dim
            ).contiguous()
            
            (output, hidden) = step_module.forward(
                lstm_input_tokens_sorted=input_tokens_sorted[:, i], 
                lstm_hidden=(last_hidden, last_cell), 
                lstm_projected_keys_textual=projected_keys_textual, 
                lstm_commands_lengths=commands_lengths, 
                lstm_projected_keys_visual=projected_keys_visual,
                tag="_lstm_step_fxn"
            )
        else:
            (output, hidden) = step_module(
                lstm_input_tokens_sorted=hidden_states["input_tokens_sorted"][:, i], 
                lstm_hidden=hidden_states["hidden"], 
                lstm_projected_keys_textual=hidden_states["projected_keys_textual"], 
                lstm_commands_lengths=hidden_states["commands_lengths"], 
                lstm_projected_keys_visual=hidden_states["projected_keys_visual"],
                tag="_lstm_step_fxn"
            )
        
        last_states = torch.cat(
                [
                    hidden[0].squeeze(dim=1),
                    hidden[1].squeeze(dim=1),
                    input_tokens_sorted,
                    commands_lengths,
                    projected_keys_visual.reshape(batch_size, -1),
                    output,
                    projected_keys_textual.reshape(batch_size, -1),
                ], dim=-1
        )
        return last_states

    return _lstm_step_fxn

def generate_compute_graph_cf(
    model, 
    max_decode_step,
    cache_results=False, 
    vocab_size=6,
    image_size=6, 
    hidden_dim=100
):
    
    
    ####################
    #
    # Input preparation.
    #
    ####################
    """
    Command Inputs.
    """
    command_world_inputs = ["commands_input", "commands_lengths"]
    command_world_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, default_value=None) 
        for name in command_world_inputs
    ]
    @GraphNode(*command_world_input_leaves, cache_results=False)
    def command_input_preparation(
        commands_input, commands_lengths,
    ):
        input_dict = {
            "commands_input": commands_input,
            "commands_lengths": commands_lengths,
        }
        # We may not need the following fields. But we leave it here in case we need these
        # to generate other inputs.
        batch_size = input_dict["commands_input"].shape[0]
        device = input_dict["commands_input"].device
        return input_dict
    
    """
    Situation Inputs.
    """
    situation_inputs = ["situations_input"]
    situation_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, 
                       default_value=None) 
        for name in situation_inputs
    ]
    @GraphNode(*situation_input_leaves, cache_results=cache_results)
    def situation_input_preparation(
        situations_input,
    ):
        return {
            "situations_input": situations_input,
        }
        
    """
    Target Inputs
    """
    target_sequence_inputs = ["target_batch", "target_lengths"]
    target_sequence_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, 
                       default_value=None) 
        for name in target_sequence_inputs
    ]
    @GraphNode(*target_sequence_input_leaves, 
               cache_results=cache_results)
    def target_sequence_input_preparation(
        target_batch, target_lengths
    ):
        return {
            "target_batch": target_batch,
            "target_lengths": target_lengths,
        }
    
    ####################
    #
    # Input encoding.
    #
    ####################
    """
    Situation Encoding.
    """
    @GraphNode(situation_input_preparation, 
               cache_results=cache_results)
    def situation_encode(input_dict):
        if isnotebook():
            return model.forward(
                situations_input=input_dict["situations_input"],
                tag="situation_encode"
            )
        else:
            return model(
                situations_input=input_dict["situations_input"],
                tag="situation_encode"
            )
    
    """
    Language Encoding.
    """
    @GraphNode(command_input_preparation, 
               cache_results=cache_results)
    def command_input_encode(input_dict):
        if isnotebook():
            return model.forward(
                commands_input=input_dict["commands_input"], 
                commands_lengths=input_dict["commands_lengths"],
                tag="command_input_encode"
            )
        else:
            return model(
                commands_input=input_dict["commands_input"], 
                commands_lengths=input_dict["commands_lengths"],
                tag="command_input_encode"
            )
    
    ####################
    #
    # Decoding.
    #
    ####################
    """
    Preparation of Decoding Data structure.
    """
    @GraphNode(command_input_encode, situation_encode, 
               target_sequence_input_preparation, 
               cache_results=cache_results)
    def decode_input_preparation(c_encode, s_encode, target_sequence):
        if isnotebook():
            hidden_states = model.forward(
                target_batch=target_sequence["target_batch"],
                target_lengths=target_sequence["target_lengths"],
                command_hidden=c_encode["command_hidden"],
                command_encoder_outputs=c_encode["command_encoder_outputs"],
                command_sequence_lengths=c_encode["command_sequence_lengths"],
                encoded_situations=s_encode,
                tag="decode_input_preparation"
            )
        else:
            hidden_states = model(
                target_batch=target_sequence["target_batch"],
                target_lengths=target_sequence["target_lengths"],
                command_hidden=c_encode["command_hidden"],
                command_encoder_outputs=c_encode["command_encoder_outputs"],
                command_sequence_lengths=c_encode["command_sequence_lengths"],
                encoded_situations=s_encode,
                tag="decode_input_preparation"
            )
        # dummy output tensor for the first time.
        batch_size = hidden_states["input_tokens_sorted"].size(0)
        return torch.cat(
                [
                    hidden_states["hidden"][0].squeeze(dim=1),
                    hidden_states["hidden"][1].squeeze(dim=1),
                    hidden_states["input_tokens_sorted"],
                    hidden_states["commands_lengths"],
                    hidden_states["projected_keys_visual"].reshape(batch_size, -1),
                    torch.zeros(batch_size, vocab_size),
                    # we need the textual key to be at last since the dimension is not interpretable.
                    hidden_states["projected_keys_textual"].reshape(batch_size, -1)
                ], dim=-1
            )
        

    hidden_layer = decode_input_preparation
    """
    Here, we set to a static bound of decoding steps.
    """
    for i in range(max_decode_step):
        f = _generate_lstm_step_fxn_cf(
            model, i, max_decode_step,
            vocab_size=vocab_size,
            image_size=image_size, 
            hidden_dim=hidden_dim
        )
        hidden_layer = GraphNode(hidden_layer,
                                 name=f"lstm_step_{i}",
                                 forward=f, cache_results=cache_results)
        
    # Do we really need this?
    # """
    # Formulating outputs.
    # """
    # @GraphNode(hidden_layer, cache_results=cache_results)
    # def output_preparation(hidden_states):
    #     hidden_states["return_lstm_output"] = torch.cat(
    #         hidden_states["return_lstm_output"], dim=0)
    #     hidden_states["return_attention_weights"] = torch.cat(
    #         hidden_states["return_attention_weights"], dim=0)
    #     
    #     decoder_output_batched = hidden_states["return_lstm_output"]
    #     context_situation = hidden_states["return_attention_weights"]
    #     decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
    #     
    #     # if model.module.auxiliary_task:
    #     if False:
    #         pass # Not implemented yet.
    #     else:
    #         target_position_scores = torch.zeros(1), torch.zeros(1)
    #        # We are not returning this as well, since it is not used...
    #    print(decoder_output_batched.shape)
    #    return decoder_output_batched.transpose(0, 1) # [batch_size, max_target_seq_length, target_vocabulary_size]
    # root = hidden_layer # TODO: removing this and continue.
    
    return hidden_layer
    
"""
Below are fastest model for regular loss calculation.
"""
    
def _generate_lstm_step_fxn(step_module, i):
    """ 
    Generate a function for a layer in lstm.
    """

    def _lstm_step_fxn(hidden_states):
        (output, hidden) = step_module(
            lstm_input_tokens_sorted=hidden_states["input_tokens_sorted"][:, i], 
            lstm_hidden=hidden_states["hidden"], 
            lstm_projected_keys_textual=hidden_states["projected_keys_textual"], 
            lstm_commands_lengths=hidden_states["commands_lengths"], 
            lstm_projected_keys_visual=hidden_states["projected_keys_visual"],
            tag="_lstm_step_fxn"
        )
        hidden_states["hidden"] = hidden
        hidden_states["return_lstm_output"] += [output.unsqueeze(0)]
        
        return hidden_states

    return _lstm_step_fxn

def generate_compute_graph(model, 
                           max_decode_step,
                           cache_results=False):
    
    
    ####################
    #
    # Input preparation.
    #
    ####################
    """
    Command Inputs.
    """
    command_world_inputs = ["commands_input", "commands_lengths"]
    command_world_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, default_value=None) 
        for name in command_world_inputs
    ]
    @GraphNode(*command_world_input_leaves, cache_results=False)
    def command_input_preparation(
        commands_input, commands_lengths,
    ):
        input_dict = {
            "commands_input": commands_input,
            "commands_lengths": commands_lengths,
        }
        # We may not need the following fields. But we leave it here in case we need these
        # to generate other inputs.
        batch_size = input_dict["commands_input"].shape[0]
        device = input_dict["commands_input"].device
        return input_dict
    
    """
    Situation Inputs.
    """
    situation_inputs = ["situations_input"]
    situation_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, 
                       default_value=None) 
        for name in situation_inputs
    ]
    @GraphNode(*situation_input_leaves, cache_results=cache_results)
    def situation_input_preparation(
        situations_input,
    ):
        return {
            "situations_input": situations_input,
        }
        
    """
    Target Inputs
    """
    target_sequence_inputs = ["target_batch", "target_lengths"]
    target_sequence_input_leaves = [
        GraphNode.leaf(name=name, use_default=True, 
                       default_value=None) 
        for name in target_sequence_inputs
    ]
    @GraphNode(*target_sequence_input_leaves, 
               cache_results=cache_results)
    def target_sequence_input_preparation(
        target_batch, target_lengths
    ):
        return {
            "target_batch": target_batch,
            "target_lengths": target_lengths,
        }
    
    ####################
    #
    # Input encoding.
    #
    ####################
    """
    Situation Encoding.
    """
    @GraphNode(situation_input_preparation, 
               cache_results=cache_results)
    def situation_encode(input_dict):
        return model(
            situations_input=input_dict["situations_input"],
            tag="situation_encode"
        )
    
    """
    Language Encoding.
    """
    @GraphNode(command_input_preparation, 
               cache_results=cache_results)
    def command_input_encode(input_dict):
        return model(
            commands_input=input_dict["commands_input"], 
            commands_lengths=input_dict["commands_lengths"],
            tag="command_input_encode"
        )
    
    ####################
    #
    # Decoding.
    #
    ####################
    """
    Preparation of Decoding Data structure.
    """
    @GraphNode(command_input_encode, situation_encode, 
               target_sequence_input_preparation, 
               cache_results=cache_results)
    def decode_input_preparation(c_encode, s_encode, target_sequence):
        return model(
            target_batch=target_sequence["target_batch"],
            target_lengths=target_sequence["target_lengths"],
            command_hidden=c_encode["command_hidden"],
            command_encoder_outputs=c_encode["command_encoder_outputs"],
            command_sequence_lengths=c_encode["command_sequence_lengths"],
            encoded_situations=s_encode,
            tag="decode_input_preparation"
        )

    hidden_layer = decode_input_preparation
    """
    Here, we set to a static bound of decoding steps.
    """
    for i in range(max_decode_step):
        f = _generate_lstm_step_fxn(model, i)
        hidden_layer = GraphNode(hidden_layer,
                                 name=f"lstm_step_{i}",
                                 forward=f, cache_results=cache_results)
    """
    Formulating outputs.
    """
    @GraphNode(hidden_layer, cache_results=cache_results)
    def output_preparation(hidden_states):
        hidden_states["return_lstm_output"] = torch.cat(
            hidden_states["return_lstm_output"], dim=0)
        decoder_output_batched = hidden_states["return_lstm_output"]
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        
        return decoder_output_batched.transpose(0, 1) # [batch_size, max_target_seq_length, target_vocabulary_size]
    
    root = output_preparation # TODO: removing this and continue.
    
    return root
    
    
class ReaSCANMultiModalLSTMCompGraph(ComputationGraph):
    def __init__(self, model,
                 max_decode_step,
                 is_cf=False,
                 cache_results=False):
        self.model = model
        if not is_cf:
            root = generate_compute_graph(
                model,
                max_decode_step,
                cache_results=cache_results,
            )
        else:
            root = generate_compute_graph_cf(
                model,
                max_decode_step,
                cache_results=cache_results,
            )
        super().__init__(root)