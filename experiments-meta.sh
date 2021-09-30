#####
#
# Dev-Ops
#
#####
# system helpers.
nvidia-htop.py
# notebook converter.
jupyter nbconvert --to python run_seq2seq.ipynb
jupyter nbconvert --to python run_mmlstm.ipynb
jupyter nbconvert --to python run_mmlstm_encoder.ipynb
jupyter nbconvert --to python run_evaluation.ipynb
jupyter nbconvert --to python run_evaluation_encoder.ipynb
jupyter nbconvert --to python generate_ReaSCAN.ipynb


#####
#
# Data Preprocessing
#
#####
# using ReaSCAN framework to generate datasets.
python generate_ReaSCAN.py \
--mode train \
--n_command_struct -1 \
--date 2021-09-28 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 200 \
--output_dir ../../data-files/ReaSCAN-Causal-ICLR-Official/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p1 \
--save_interal 50 \
--seed 42
# Note that our novel direction split is directly adapted from gSCAN
# since ReaSCAN is not supporting that yet.
# We still provide scripts to split for novel direction, but consider
# not to use it.

# for new length tests, we need to slightly modify the dataset
# generation script.
python generate_ReaSCAN.py \
--mode train \
--n_command_struct -1 \
--date 2021-09-28 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 1200 \
--output_dir ../../data-files/ReaSCAN-Causal-ICLR-Official-novel-length/ \
--include_relation_distractor \
--include_attribute_distractor \
--include_isomorphism_distractor \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p1 \
--save_interal 50 \
--seed 42 \
--simple_command
# Note that we change --per_command_world_target_count now up to 1200
# this is because we only allow "walk" as our verb, and we don't allow
# adverb in this case for simplicity.
# --simple_command is for generating these commands.

# after you generate the dataset, you need to use the splitter to
# make sure you have testing sets splitted correctly.
# ReaSCAN_splitter.ipynb


#####
#
# Training
#
#####

# Counterfactual training for novel attributes.
CUDA_VISIBLE_DEVICES=0 python run_mmlstm_encoder.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-attribute/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=88 \
--learning_rate 0.002 \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 25 \
--cf_sample_p 1.0 \
--cf_loss_weight 1.0 \
--intervene_position last_hidden \
--is_wandb
# You can remove these flags to exclude losses.
# --include_task_loss
# --include_cf_loss
# --include_cf_auxiliary_loss
# You may add this option for restrict sampling of counterfactual example pairs.
# --restrict_sampling by_attribute

# Counterfactual training for novel action sequence (direction/length).
CUDA_VISIBLE_DEVICES=0 python run_mmlstm.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/gSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=88 \
--learning_rate 0.002 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--cf_sample_p 1.0 \
--cf_loss_weight 1.0 \
--is_wandb
# You can remove these flags to exclude losses.
# --include_task_loss
# --include_cf_loss
# --include_cf_auxiliary_loss
# You can the data directory for the other task.
# --data_directory
# You may add this option for restrict sampling of counterfactual example pairs.
# --restrict_sampling by_direction
# --restrict_sampling by_length

# For regular training, you can simply run our baseline trainer.
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-attribute/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=88 \
--learning_rate 0.002 \
--is_wandb
# You can the data directory for the other task.
# --data_directory
