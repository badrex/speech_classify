#!/usr/bin/env bash

# specify which GPU to work on ...
export CUDA_VISIBLE_DEVICES=1

nvidia-smi


# obtain the directory the bash script is stored in
DIR=$(cd $(dirname $0); pwd)
python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
# python -u $DIR/../nn_train/nn_train_baseline.py $DIR/../config_files/config_file_baseline_mfsc_sbc.yml
