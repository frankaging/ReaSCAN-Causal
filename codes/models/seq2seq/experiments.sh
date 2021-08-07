# train seq2seq model with antra as backend
CUDA_VISIBLE_DEVICES=0 python run_seq2seq.py \
--mode=train \
--max_decoding_steps=120 \
--max_testing_examples=2000 \
--data_directory=../../../data-files/ReaSCAN-Simple/ \
--input_vocab_path=input_vocabulary.txt \
--target_vocab_path=target_vocabulary.txt \
--attention_type=bahdanau \
--no_auxiliary_task \
--conditional_attention \
--output_directory=../../../ \
--training_batch_size=2000 \
--max_training_iterations=200000 \
--seed=44