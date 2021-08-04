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
            hidden_states["input_tokens_sorted"][:, i], 
            hidden_states["hidden"], 
            hidden_states["projected_keys_textual"], 
            hidden_states["commands_lengths"], 
            hidden_states["projected_keys_visual"],
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
        input_dict = {
            "situations_input": situations_input,
        }
        return input_dict
        
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
        input_dict = {
            "target_batch": target_batch,
            "target_lengths": target_lengths,
        }
        return input_dict
    
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
        encoded_image = model.situation_encoder(
            input_images=input_dict["situations_input"]
        )
        return encoded_image
    
    """
    Language Encoding.
    """
    @GraphNode(command_input_preparation, 
               cache_results=cache_results)
    def command_input_encode(input_dict):
        hidden, encoder_outputs = model.encoder(
            input_batch=input_dict["commands_input"], 
            input_lengths=input_dict["commands_lengths"],
        )
        output_dict = {
            "command_hidden" : hidden,
            "command_encoder_outputs" : encoder_outputs["encoder_outputs"],
            "command_sequence_lengths" : encoder_outputs["sequence_lengths"],
        }
        return output_dict
    
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
        """
        The decoding step can be represented as:
        h_T = f(h_T-1, C)
        where h_i is the recurring hidden states, and C
        is the static state representations.
        
        In this function, we want to abstract the C.
        """
        
        initial_hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(c_encode["command_hidden"])))
        
        """
        Renaming.
        """
        input_tokens, input_lengths = target_sequence["target_batch"], target_sequence["target_lengths"]
        init_hidden = initial_hidden
        encoded_commands = c_encode["command_encoder_outputs"]
        commands_lengths = c_encode["command_sequence_lengths"]
        encoded_situations = s_encode
        
        """
        Reshaping as well as getting the context-guided attention weights.
        """
        # Deprecated. We don't need to sort anymore.
        # Sort the sequences by length in descending order
        # batch_size, max_time = input_tokens.size()
        # input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        # input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        # input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        # initial_h, initial_c = init_hidden
        # hidden = (initial_h.index_select(dim=1, index=perm_idx),
        #           initial_c.index_select(dim=1, index=perm_idx))
        # encoded_commands = encoded_commands.index_select(dim=1, index=perm_idx)
        # commands_lengths = torch.tensor(commands_lengths, device=device)
        # commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)
        # encoded_situations = encoded_situations.index_select(dim=0, index=perm_idx)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]
        
        return {
            "return_lstm_output":[],
            "return_attention_weights":[],
            "hidden":init_hidden,
            "input_tokens_sorted":input_tokens,
            "projected_keys_textual":projected_keys_textual,
            "commands_lengths":commands_lengths,
            "projected_keys_visual":projected_keys_visual,
            "seq_lengths":input_lengths,
        }

    hidden_layer = decode_input_preparation
    """
    Here, we set to a static bound of decoding steps.
    """
    for i in range(max_decode_step):
        f = _generate_lstm_step_fxn(model.attention_decoder.forward_step, i)
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
        
        if model.auxiliary_task:
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