# system helpers.
nvidia-htop.py

# notebook converter.
jupyter nbconvert --to python run_seq2seq.ipynb
jupyter nbconvert --to python run_counterfactual.ipynb
jupyter nbconvert --to python run_evaluation.ipynb
jupyter nbconvert --to python generate_ReaSCAN.ipynb

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








