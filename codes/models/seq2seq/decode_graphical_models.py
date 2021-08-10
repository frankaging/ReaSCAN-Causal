import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import json
from model import *
from ReaSCAN_dataset import *
import torch.nn.functional as F
import torch
from antra.antra import *

def _generate_lstm_step_fxn(step_module, i):
    """ 
    Generate a function for a layer in lstm.
    """

    def _lstm_step_fxn(hidden_states):
        (output, hidden, context_situation, attention_weights_commands,
         attention_weights_situations) = step_module(
            lstm_input_tokens_sorted=hidden_states["input_tokens_sorted"][:, i], 
            lstm_hidden=hidden_states["hidden"], 
            lstm_projected_keys_textual=hidden_states["projected_keys_textual"], 
            lstm_commands_lengths=hidden_states["commands_lengths"], 
            lstm_projected_keys_visual=hidden_states["projected_keys_visual"],
            tag="_lstm_step_fxn"
        )
        hidden_states["hidden"] = hidden
        hidden_states["return_lstm_output"] += [output.unsqueeze(0)]
        hidden_states["return_attention_weights"] += [attention_weights_situations.unsqueeze(0)]
        
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
        hidden_states["return_attention_weights"] = torch.cat(
            hidden_states["return_attention_weights"], dim=0)
        
        decoder_output_batched = hidden_states["return_lstm_output"]
        context_situation = hidden_states["return_attention_weights"]
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        
        # if model.module.auxiliary_task:
        if False:
            pass # Not implemented yet.
        else:
            target_position_scores = torch.zeros(1), torch.zeros(1)
            # We are not returning this as well, since it is not used...
        
        return decoder_output_batched.transpose(0, 1) # [batch_size, max_target_seq_length, target_vocabulary_size]
    
    root = output_preparation # TODO: removing this and continue.
    
    return root
    
class ReaSCANMultiModalLSTMCompGraph(ComputationGraph):
    def __init__(self, model,
                 max_decode_step,
                 cache_results=False):
        self.model = model
        root = generate_compute_graph(
            model,
            max_decode_step,
        )

        super().__init__(root)