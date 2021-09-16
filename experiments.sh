#####
#
# Dev-Ops
#
#####
# system helpers.
nvidia-htop.py
# notebook converter.
jupyter nbconvert --to python run_seq2seq.ipynb
jupyter nbconvert --to python run_counterfactual.ipynb
jupyter nbconvert --to python run_counterfactual_encoder.ipynb
jupyter nbconvert --to python run_evaluation.ipynb
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
--date 2021-09-09 \
--grid_size 6 \
--n_object_max 13 \
--per_command_world_retry_max 500 \
--per_command_world_target_count 200 \
--output_dir ../../data-files/ReaSCAN-Causal-new-attribute/ \
--include_random_distractor \
--full_relation_probability 1.0 \
--command_pattern p1 \
--save_interal 50 \
--seed 42


#####
#
# Training
#
#####
# train MM-LSTM baselines
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
--seed=42 \
--learning_rate 0.002 \
--is_wandb

CUDA_VISIBLE_DEVICES=4 python run_seq2seq.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-action-length/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--is_wandb

CUDA_VISIBLE_DEVICES=1 python run_seq2seq.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--is_wandb

CUDA_VISIBLE_DEVICES=7 python run_seq2seq.py \
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
--seed=42 \
--learning_rate 0.002 \
--is_wandb

# train GCN+LSTM model (SoTA model by the time we submit our paper)
# run on CPU is fine with this model.
python main_model.py \
--data_dir ./data-files/ReaSCAN-novel-attribute/ \
--seed 42 \
--is_wandb

python main_model.py \
--data_dir ./data-files/ReaSCAN-novel-action-length/ \
--seed 42 \
--is_wandb

python main_model.py \
--data_dir ./data-files/ReaSCAN-novel-direction/ \
--seed 42 \
--is_wandb

python main_model.py \
--data_dir ./data-files/gSCAN-novel-direction/ \
--seed 42 \
--is_wandb

# train our action sequence counterfactual model
# with probe loss + cf loss
CUDA_VISIBLE_DEVICES=0 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-action-length/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

CUDA_VISIBLE_DEVICES=2 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

CUDA_VISIBLE_DEVICES=2 python run_counterfactual.py \
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
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

# train our action sequence counterfactual model
# with probe loss only
CUDA_VISIBLE_DEVICES=1 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-action-length/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

CUDA_VISIBLE_DEVICES=7 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

CUDA_VISIBLE_DEVICES=1 python run_counterfactual.py \
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
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50

# train our action sequence counterfactual model
# with probe loss + cf loss
# BUT without seen testing composites during training
CUDA_VISIBLE_DEVICES=1 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-action-length/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--restrict_sampling by_length

CUDA_VISIBLE_DEVICES=9 python run_counterfactual.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=200 \
--max_training_iterations=100000 \
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--restrict_sampling by_direction

CUDA_VISIBLE_DEVICES=6 python run_counterfactual.py \
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
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--is_wandb \
--include_task_loss \
--include_cf_loss \
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--restrict_sampling by_direction

# train our novel attribute counterfactual model
# these are intervening on the embedding directly.
# these are expected to work unless there is a bug!
CUDA_VISIBLE_DEVICES=1 python run_counterfactual_encoder.py \
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
--seed=42 \
--learning_rate 0.002 \
--print_every 25 \
--include_task_loss \
--include_cf_loss \
--intervene_position embedding \
--is_wandb


#####
#
# Evaluation
#
#####
# evaluate MM-LSTM models on compositional splits.
CUDA_VISIBLE_DEVICES=1 python run_evaluation.py \
--mode=test \
--seed=42 \
--data_directory=../../../data-files/ReaSCAN-novel-attribute/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--max_decoding_steps=120 \
--resume_from_file=../../../mmlstm_ReaSCAN-novel-attribute_seed_42_lr_0.002/ \
--splits=dev \
--counterfactual_evaluate \
--max_testing_examples=2000 \
--no_cuda

CUDA_VISIBLE_DEVICES=1 python run_evaluation.py \
--mode=test \
--seed=42 \
--data_directory=../../../data-files/ReaSCAN-novel-attribute/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--max_decoding_steps=120 \
--resume_from_file=../../../mmlstm_ReaSCAN-novel-attribute_seed_42_lr_0.002/ \
--splits=test \
--counterfactual_evaluate \
--max_testing_examples=2000 \
--no_cuda

# evaluate with counterfactually trained models
CUDA_VISIBLE_DEVICES=1 python run_evaluation.py \
--mode=test \
--seed=42 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--max_decoding_steps=120 \
--resume_from_file=../../../counterfactual_ReaSCAN-novel-direction_seed_42_lr_0.002_attr_-2_size_50_cf_loss_True_aux_loss_True/ \
--splits=dev \
--counterfactual_evaluate \
--intervene_dimension_size 50 \
--max_testing_examples=1000

CUDA_VISIBLE_DEVICES=1 python run_evaluation.py \
--mode=test \
--seed=42 \
--data_directory=../../../data-files/ReaSCAN-novel-direction/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--max_decoding_steps=120 \
--resume_from_file=../../../counterfactual_ReaSCAN-novel-direction_seed_42_lr_0.002_attr_-2_size_50_cf_loss_True_aux_loss_True/ \
--splits=test \
--counterfactual_evaluate \
--intervene_dimension_size 50 \
--max_testing_examples=2000 \
--no_cuda

# evaluate with unseen splits (we jsut need to turn off the cf accuracy flag)
CUDA_VISIBLE_DEVICES=1 python run_evaluation.py \
--mode=test \
--seed=42 \
--data_directory=../../../data-files/ReaSCAN-novel-attribute/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--max_decoding_steps=120 \
--resume_from_file=../../../mmlstm_ReaSCAN-novel-attribute_seed_42_lr_0.002/ \
--splits=new_size \
--intervene_dimension_size 25 \
--max_testing_examples=2000 \
--no_cuda

# evaluate GCN+LSTM models (counterfactual evaluation is not supported here).
CUDA_VISIBLE_DEVICES=1 python eval_best_model.py \
--load ../../../gcnlstm_ReaSCAN-novel-action-length_seed_42_lr_0.0008/checkpoint_force.80th.tar \
--data_dir ./data-files/ReaSCAN-novel-action-length/ \
--seed 42 \
--test_split dev

CUDA_VISIBLE_DEVICES=1 python eval_best_model.py \
--load ../../../gcnlstm_ReaSCAN-novel-action-length_seed_42_lr_0.0008/checkpoint_force.80th.tar \
--data_dir ./data-files/ReaSCAN-novel-action-length/ \
--seed 42 \
--test_split test



