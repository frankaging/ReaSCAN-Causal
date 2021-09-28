# Change random seeds to 66 and 44.

# Baselines
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py \
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
--is_wandb

CUDA_VISIBLE_DEVICES=1 python run_seq2seq.py \
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

CUDA_VISIBLE_DEVICES=2 python run_seq2seq.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-length/ \
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

# Novel attribute with different conditions.
CUDA_VISIBLE_DEVICES=3 python run_mmlstm_encoder.py \
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
--include_cf_auxiliary_loss \
--intervene_dimension_size 25 \
--cf_sample_p 1.0 \
--cf_loss_weight 1.0 \
--intervene_position last_hidden \
--is_wandb

CUDA_VISIBLE_DEVICES=5 python run_mmlstm_encoder.py \
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

CUDA_VISIBLE_DEVICES=9 python run_mmlstm_encoder.py \
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
--restrict_sampling by_attribute \
--is_wandb

# Novel direction
CUDA_VISIBLE_DEVICES=7 python run_mmlstm.py \
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
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--cf_sample_p 1.0 \
--cf_loss_weight 1.0 \
--is_wandb

CUDA_VISIBLE_DEVICES=4 python run_mmlstm.py \
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

CUDA_VISIBLE_DEVICES=1 python run_mmlstm.py \
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
--is_wandb \
--restrict_sampling by_direction

# Novel action length
CUDA_VISIBLE_DEVICES=6 python run_mmlstm.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-length/ \
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
--include_cf_auxiliary_loss \
--intervene_dimension_size 50 \
--cf_sample_p 1.0 \
--cf_loss_weight 1.0 \
--is_wandb

CUDA_VISIBLE_DEVICES=7 python run_mmlstm.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-length/ \
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

CUDA_VISIBLE_DEVICES=8 python run_mmlstm.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-novel-length/ \
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
--is_wandb \
--restrict_sampling by_length

