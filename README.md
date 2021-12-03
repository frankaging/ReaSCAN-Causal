# Interchange Intervention Training (IIT) for Compositional Reasoning in Language Grounding
Towards Solving ReaSCAN Using Counterfactually Trained Neural Models

## Release Notes
* **10/01/2021**: We are preparing to release our code.

## Contents
* [Citation](#citation)
* [Dataset](#dataset)
* [Models](#models)
* [Training](#training)
* [Other files](#other-files)
* [License](#license)

## Citation
If you use this repository, please cite the following two papers: [paper for interchange intervention training](https://arxiv.org/abs/2112.00826), and [paper for the ReaSCAN dataset](https://arxiv.org/abs/2109.08994).
```stex
  @article{geiger-etal-2021-iit,
        title={Inducing Causal Structure for Interpretable Neural Networks}, 
        author={Geiger, Atticus and Wu, Zhengxuan and Lu, Hanson and Rozner, Josh and Kreiss, Elisa and Icard, Thomas and Goodman, Noah D. and Potts, Christopher},
        year={2021},
        eprint={2112.00826},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
  }

  @article{wu-etal-2021-reascan,
    title={Rea{SCAN}: Compositional Reasoning in Language Grounding},
    author={Wu, Zhengxuan and Kreiss, Elisa and Ong, Desmond C. and Potts, Christopher},
    journal={NeurIPS 2021 Datasets and Benchmarks Track},
    url={https://openreview.net/forum?id=Rtquf4Jk0jN},
    year={2021}}
```

## Dataset

### Off-the-shelf regenerated ReaSCAN
We use ReaSCAN framework to generate datasets for different experiments. In addition, we also use the `novel direction` split provided by [gSCAN](https://github.com/LauraRuis/groundedSCAN). For all the datasets we use, you can download them off-the-shelves from `ReaSCAN-Causal.zip` in this main folder.

### Regenerate ReaSCAN that we use
We also provide you the full-fledged adpated ReaSCAN framework for you to generate datasets that we use in our experiments. The scripts we use are all in the `experiments-meta.sh` script, here is one example:
```bash
cd codes/Reason-SCAN/code/dataset/

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
```

## Models

### Multimodal LSTM

This model is published with gSCAN [in this paper](https://arxiv.org/abs/2003.05161) from [this repo](https://github.com/LauraRuis/multimodal_seq2seq_gSCAN). You can refer to their repo for details about the model. Here, we already adapt interface changes that are needed to run with ReaSCAN.

## Training

We provide you training scripts for different setttings, including regular training, counterfactual training and multi-task training. We also provide you in-depth evaluation scripts that you can use to evaluate your results. You can refer to our provided scripts to see how we run our experiments in `experiments-meta.sh` and `experiments.sh`

Regular training,
```bash
cd codes/models/seq2seq/

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
```

Counterfactual training for POSITION variables,
```bash
cd codes/models/seq2seq/

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
```

Counterfactual training for ATTRIBUTE variables,
```bash
cd codes/models/seq2seq/

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

# Counterfactual training for novel length and direction.
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
```

## Other files
We also provide other helper scripts help you to visualize datasets, split datasets and etc..
* `codes/models/seq2seq/ReaSCAN_splitter.ipynb` for splitting the datasets.
* `codes/models/seq2seq/abstraction_graphical_model_demo.ipynb` for demonstration for our abstract models.

## License

ReaSCAN has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).


