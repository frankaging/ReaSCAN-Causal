import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.cnn_model import DownSamplingConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import Attention
from seq2seq.seq2seq_model import BahdanauAttentionDecoderRNN

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class Model(nn.Module):

    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, target_position_size: int, 
                 intervene_dimension_size: int, **kwargs):
        super(Model, self).__init__()

        logger.warning(f"Model output directory: {output_directory}")
        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.visual_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=decoder_hidden_size,
                                          hidden_size=decoder_hidden_size)

        self.auxiliary_task = auxiliary_task
        if auxiliary_task:
            self.auxiliary_loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)
        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.enc_hidden_to_dec_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.textual_attention = Attention(key_size=encoder_hidden_size, query_size=decoder_hidden_size,
                                           hidden_size=decoder_hidden_size)

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]
        self.attention_type = attention_type
        if attention_type == "bahdanau":
            self.attention_decoder = BahdanauAttentionDecoderRNN(hidden_size=decoder_hidden_size,
                                                                 output_size=target_vocabulary_size,
                                                                 num_layers=num_decoder_layers,
                                                                 dropout_probability=decoder_dropout_p,
                                                                 padding_idx=target_pad_idx,
                                                                 textual_attention=self.textual_attention,
                                                                 visual_attention=self.visual_attention,
                                                                 conditional_attention=conditional_attention)
        else:
            raise ValueError("Unknown attention type {} specified.".format(attention_type))
        self.decoder_hidden_init = nn.Linear(3*decoder_hidden_size, decoder_hidden_size)
        
        # We use these two variables to encode target and agent position offsets.
        # Be careful that this is NOT the x, y for the target, IT IS the relative 
        # x, y for the target w.r.t. the agent.
        self.target_position_decoder_x = nn.Linear(intervene_dimension_size, target_position_size)
        self.target_position_decoder_y = nn.Linear(intervene_dimension_size, target_position_size)
        
        self.size_decoder = nn.Linear(intervene_dimension_size, 2)
        self.color_decoder = nn.Linear(intervene_dimension_size, 4)
        self.shape_decoder = nn.Linear(intervene_dimension_size, 3)
            
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.loss_criterion = nn.NLLLoss(ignore_index=target_pad_idx)
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = torch.cat([input_tensor, torch.zeros(batch_size, device=input_tensor.device, dtype=torch.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    def get_metrics(self, target_scores: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    @staticmethod
    def get_auxiliary_accuracy(target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            predicted_targets = target_scores.max(dim=1)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long().sum().data.item()
            accuracy = 100. * equal_targets / len(targets)
        return accuracy

    def get_loss(self, target_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        
        targets = self.remove_start_of_sequence(targets)

        # Calculate the loss.
        _, _, vocabulary_size = target_scores.size()
        target_scores_2d = target_scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def get_auxiliary_loss(self, auxiliary_scores_target: torch.Tensor, target_target_positions: torch.Tensor):
        target_loss = self.auxiliary_loss_criterion(auxiliary_scores_target, target_target_positions.view(-1))
        return target_loss
    
    def get_cf_auxiliary_loss(self, target_position_scores, target_positions):
        loss = self.loss_criterion(target_position_scores, target_positions.view(-1))
        return loss
    
    def get_cf_auxiliary_metrics(self, target_position_scores, target_positions):
        predicted_target_positions = target_position_scores.max(dim=-1)[1]
        equal_targets = torch.eq(target_positions.data, predicted_target_positions.data).long()
        match_sum_per_batch = equal_targets.sum()
        batch_size = target_position_scores.size(0)
        exact_match = 100. * match_sum_per_batch / batch_size
        return exact_match
    
    def auxiliary_task_forward(self, output_scores_target_pos: torch.Tensor) -> torch.Tensor:
        assert self.auxiliary_task, "Please set auxiliary_task to True if using it."
        batch_size, _ = output_scores_target_pos.size()
        output_scores_target_pos = F.log_softmax(output_scores_target_pos, -1)
        return output_scores_target_pos

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched, _, context_situation = self.attention_decoder(input_tokens=target_batch,
                                                                              input_lengths=target_lengths,
                                                                              init_hidden=initial_hidden,
                                                                              encoded_commands=encoded_commands,
                                                                              commands_lengths=command_lengths,
                                                                              encoded_situations=encoded_situations)
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        return decoder_output_batched, context_situation

    def forward(self, 
                # raw inputs
                commands_input=None, 
                commands_embedding=None,
                commands_lengths=None, 
                situations_input=None,
                target_batch=None,
                target_lengths=None,
                # intermediate layers
                command_hidden=None,
                command_encoder_outputs=None,
                command_sequence_lengths=None,
                encoded_situations=None,
                # recurrent layers
                lstm_input_tokens_sorted=None,
                lstm_hidden=None,
                lstm_projected_keys_textual=None,
                lstm_commands_lengths=None,
                lstm_projected_keys_visual=None,
                # loss
                loss_target_scores=None,
                loss_target_batch=None,
                # auxiliary related hidden states
                auxiliary_hidden=None,
                loss_pred_target_auxiliary=None,
                loss_true_target_auxiliary=None,
                # others
                is_best=None,
                accuracy=None,
                exact_match=None,
                cf_auxiliary_task_tag=None,
                tag="default",
               ):
        if tag == "situation_encode":
            return self.situation_encoder(situations_input)
        elif tag == "command_input_encode_embedding":
            return self.encoder(
                input_batch=commands_input, 
                tag="encode_embedding"
            )
        elif tag == "command_input_encode_no_dict_with_embedding":
            return self.encoder(
                input_embeddings=commands_embedding, 
                input_lengths=commands_lengths,
                tag="forward_with_embedding"
            )
        elif tag == "command_input_encode_no_dict":
            return self.encoder(
                input_batch=commands_input, 
                input_lengths=commands_lengths,
                tag="forward"
            )
        elif tag == "command_input_encode":
            hidden, encoder_outputs = self.encoder(
                input_batch=commands_input, 
                input_lengths=commands_lengths,
            )
            return {
                "command_hidden" : hidden,
                "command_encoder_outputs" : encoder_outputs["encoder_outputs"],
                "command_sequence_lengths" : encoder_outputs["sequence_lengths"],
            }
        elif tag == "decode_input_preparation":
            """
            The decoding step can be represented as:
            h_T = f(h_T-1, C)
            where h_i is the recurring hidden states, and C
            is the static state representations.

            In this function, we want to abstract the C.
            """
            # this step cannot be distributed as the init state is shared across GPUs.
            initial_hidden = self.attention_decoder.initialize_hidden(
                self.tanh(self.enc_hidden_to_dec_hidden(command_hidden)))
            """
            Renaming.
            """
            input_tokens, input_lengths = target_batch, target_lengths
            init_hidden = initial_hidden
            encoded_commands = command_encoder_outputs
            commands_lengths = command_sequence_lengths

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
            projected_keys_visual = self.visual_attention.key_layer(
                encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
            projected_keys_textual = self.textual_attention.key_layer(
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
        elif tag == "initialize_hidden":
            return self.attention_decoder.initialize_hidden(
                self.tanh(self.enc_hidden_to_dec_hidden(command_hidden)))
        elif tag == "projected_keys_visual":
            return self.visual_attention.key_layer(
                encoded_situations
            )  # [batch_size, situation_length, dec_hidden_dim]
        elif tag == "projected_keys_textual":
            return self.textual_attention.key_layer(
                command_encoder_outputs
            )  # [max_input_length, batch_size, dec_hidden_dim]
        elif tag == "_lstm_step_fxn":
            return self.attention_decoder.forward_step(
                lstm_input_tokens_sorted,
                lstm_hidden,
                lstm_projected_keys_textual,
                lstm_commands_lengths,
                lstm_projected_keys_visual,
            )
        elif tag == "lstm_hidden_init":
            
            # encode shape world and command.
            command_hidden, encoder_outputs = self.encoder(
                input_batch=commands_input, 
                input_lengths=commands_lengths,
            ) # this hidden is pre-hidden, not formalized yet.
            encoded_image = self.situation_encoder(situations_input)
            # For efficiency
            projected_keys_visual = self.visual_attention.key_layer(
                encoded_image)  # [batch_size, situation_length, dec_hidden_dim]
            projected_keys_textual = self.textual_attention.key_layer(
                encoder_outputs["encoder_outputs"])  # [max_input_length, batch_size, dec_hidden_dim]
            
            # intialize the hidden states.
            initial_hidden = self.attention_decoder.initialize_hidden(
                self.tanh(self.enc_hidden_to_dec_hidden(command_hidden)))
            initial_hidden, initial_cell = initial_hidden
            
            # further contextualize hidden states.
            context_command, _ = self.attention_decoder.textual_attention(
                queries=initial_hidden, projected_keys=projected_keys_textual,
                values=projected_keys_textual, memory_lengths=encoder_outputs["sequence_lengths"]
            )
            # update hidden states here with conditioned image
            batch_size, image_num_memory, _ = projected_keys_textual.size()
            situation_lengths = torch.tensor(
                [image_num_memory for _ in range(batch_size)]
            ).long().to(device=commands_input.device)
            if self.attention_decoder.conditional_attention:
                queries = torch.cat([initial_hidden, context_command], dim=-1)
                queries = self.attention_decoder.tanh(self.attention_decoder.queries_to_keys(queries))
            else:
                queries = initial_hidden
            context_situation, _ = self.attention_decoder.visual_attention(
                queries=queries, projected_keys=projected_keys_visual,
                values=projected_keys_visual, memory_lengths=situation_lengths.unsqueeze(dim=-1))
            # context : [batch_size, 1, hidden_size]
            # attention_weights : [batch_size, 1, max_input_length]
            
            # get the initial hidden states
            context_embedding = torch.cat([initial_hidden, context_command, context_situation], dim=-1)
            hidden_init = self.decoder_hidden_init(context_embedding)
            
            return (hidden_init.transpose(0, 1), initial_cell.transpose(0, 1))
        elif tag == "_lstm_step_fxn_ib":
            if lstm_input_tokens_sorted == None:
                # this code path is reserved for intervention ONLY!
                curr_hidden, curr_cell = lstm_hidden
                output = self.attention_decoder.hidden_to_output(
                    curr_hidden.transpose(0, 1)
                ) # [batch_size, 1, output_size]
                output = output.squeeze(dim=1)   # [batch_size, output_size]
                
                return lstm_hidden, output
            else:
                # hidden.
                embedded_input = self.attention_decoder.embedding(lstm_input_tokens_sorted)
                embedded_input = self.attention_decoder.dropout(embedded_input)
                embedded_input = embedded_input.unsqueeze(1)  # [batch_size, 1, hidden_size]
                
                curr_hidden, curr_cell = lstm_hidden
                _, hidden = self.attention_decoder.lstm_cell(
                    embedded_input, lstm_hidden
                )
                
                # action.
                output = self.attention_decoder.hidden_to_output(
                    hidden[0].transpose(0, 1)
                ) # [batch_size, 1, output_size]
                output = output.squeeze(dim=1)   # [batch_size, output_size]
                
                return hidden, output
        elif tag == "loss":
            return self.get_loss(loss_target_scores, loss_target_batch)
        elif tag == "update_state":
            return self.update_state(
                is_best=is_best, accuracy=accuracy, exact_match=exact_match,
            )
        elif tag == "get_metrics":
            return self.get_metrics(
                loss_target_scores, loss_target_batch
            )
        elif tag == "cf_auxiliary_task":
            assert cf_auxiliary_task_tag != None
            assert auxiliary_hidden != None
            if cf_auxiliary_task_tag == "x":
                auxiliary_pred = self.target_position_decoder_x(auxiliary_hidden)
            elif cf_auxiliary_task_tag == "y":
                auxiliary_pred = self.target_position_decoder_y(auxiliary_hidden)
            elif cf_auxiliary_task_tag == "size":
                auxiliary_pred = self.size_decoder(auxiliary_hidden)
            elif cf_auxiliary_task_tag == "color":
                auxiliary_pred = self.color_decoder(auxiliary_hidden)
            elif cf_auxiliary_task_tag == "shape":
                auxiliary_pred = self.shape_decoder(auxiliary_hidden)
            else:
                assert False
            return auxiliary_pred
        elif tag == "cf_auxiliary_task_loss":
            return self.get_cf_auxiliary_loss(loss_pred_target_auxiliary, loss_true_target_auxiliary)
        elif tag == "cf_auxiliary_task_metrics":
            return self.get_cf_auxiliary_metrics(loss_pred_target_auxiliary, loss_true_target_auxiliary)
        else:
            encoder_output = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                               situations_input=situations_input)
            decoder_output, context_situation = self.decode_input_batched(
                target_batch=target_batch, target_lengths=target_lengths, initial_hidden=encoder_output["hidden_states"],
                encoded_commands=encoder_output["encoded_commands"]["encoder_outputs"], command_lengths=commands_lengths,
                encoded_situations=encoder_output["encoded_situations"])
            if self.auxiliary_task:
                target_position_scores = self.auxiliary_task_forward(context_situation)
            else:
                target_position_scores = torch.zeros(1), torch.zeros(1)
            return (decoder_output.transpose(0, 1),  # [batch_size, max_target_seq_length, target_vocabulary_size]
                    target_position_scores)

    def update_state(self, is_best: bool, accuracy=None, exact_match=None) -> {}:
        self.trained_iterations += 1
        if is_best:
            self.best_exact_match = exact_match
            self.best_accuracy = accuracy
            self.best_iteration = self.trained_iterations

    def load_model(self, device, path_to_checkpoint: str, strict=True) -> dict:
        checkpoint = torch.load(path_to_checkpoint, map_location=device)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path