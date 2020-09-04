#! /usr/bin/env python3
# coding: utf-8

import os
import yaml
import pprint
import sys
import collections

# to get time model was trained
from datetime import datetime
import pytz

# NOTE: import torch before pandas, otherwise segementation fault error occurs
# The couse of this problem is UNKNOWN, and not solved yet
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.metrics import balanced_accuracy_score

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torch.nn as nn
import torch.optim as optim

from nn_speech_models import *
import train_utils

# Training Routine

# obtain yml config file from cmd line and print out content
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
print('YML configuration file content:')
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(config_args)


# get time in CET timezone
current_time = datetime.now(pytz.timezone('Europe/Amsterdam'))
current_time_str = current_time.strftime("%d%m%Y_%H_%M_%S") # YYYYMMDD HH:mm:ss
#print(current_time_str)


# make a model str ID, this will be used to save model on desk
config_args['model_str'] = '_'.join(str(_var) for _var in
    [
        current_time_str,
        config_args['experiment_name'],
        config_args['encoder_arch']['encoder_model'],
        config_args['classifier_arch']['input_dim'],
        config_args['classifier_arch']['hidden_dim'],
        config_args['input_signal_params']['feature_type']
    ]
)


# make the dir str where the model will be stored
if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'], config_args['model_str']
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))

# if dir does not exits on desk, make it
train_utils.handle_dirs(config_args['model_save_dir'])


 # Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))


# Set seed for reproducibility
train_utils.set_seed_everywhere(config_args['seed'], config_args['cuda'])


##### HERE IT ALL STARTS ...
# source dataset  & featurizer ...
src_speech_df = pd.read_csv(config_args['source_speech_metadata'],
    delimiter="\t", encoding='utf-8')

src_label_set=config_args['source_language_set'].split()

# get only target labels and more than 3.0 seconds
src_speech_df = src_speech_df[
    (src_speech_df.language.isin(src_label_set)) &
    (src_speech_df.duration>3.0)
]#.sample(n=500, random_state=1)

src_speech_featurizer = SpeechFeaturizer(
    data_dir=config_args['source_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['source_language_set'].split(), # split str into list of str
    num_frames=config_args['input_signal_params']['num_frames'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['encoder_arch']['spectral_dim'],
    start_index=config_args['input_signal_params']['start_index'],
    end_index=config_args['input_signal_params']['end_index']
)

print('Source SpeechFeaturizer was initialized: ',
    src_speech_featurizer.index2label)

# target dataset & featurizer ...
tgt_speech_df = pd.read_csv(config_args['target_speech_metadata'],
    delimiter="\t", encoding='utf-8')

tgt_label_set=config_args['target_language_set'].split()

# get only target labels and more than 3.0 seconds
tgt_speech_df = tgt_speech_df[
    (tgt_speech_df.language.isin(tgt_label_set)) &
    (tgt_speech_df.duration>3.0)
]#.sample(n=500, random_state=1)


tgt_speech_featurizer = SpeechFeaturizer(
    data_dir=config_args['target_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['target_language_set'].split(), # split str into list of str
    num_frames=config_args['input_signal_params']['num_frames'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['encoder_arch']['spectral_dim'],
    start_index=config_args['input_signal_params']['start_index'],
    end_index=config_args['input_signal_params']['end_index']
)

print('Target SpeechFeaturizer was initialized: ',
    tgt_speech_featurizer.index2label)


# src & tgt data loader ...
src_speech_dataset = SpeechDataset(src_speech_df, src_speech_featurizer)
tgt_speech_dataset = SpeechDataset(tgt_speech_df, tgt_speech_featurizer)


# initialize speech encoder
if config_args['encoder_arch']['encoder_model'] == 'ConvEncoder':
    nn_speech_encoder = ConvSpeechEncoder(
        spectral_dim=config_args['encoder_arch']['spectral_dim'],
        num_channels=config_args['encoder_arch']['num_channels'],
        filter_sizes=config_args['encoder_arch']['filter_sizes'],
        stride_steps=config_args['encoder_arch']['stride_steps'],
        pooling_type=config_args['encoder_arch']['pooling_type'],
        dropout_frames=config_args['encoder_arch']['frame_dropout'],
        dropout_spectral_features=config_args['encoder_arch']['feature_dropout'],
        signal_dropout_prob=config_args['encoder_arch']['signal_dropout_prob']
    )

else:
    raise NotImplementedError


# initialize main task classifier ...
nn_task_classifier = FeedforwardClassifier(
    num_classes= config_args['classifier_arch']['num_classes'], # or len(label_set)
    input_dim=config_args['classifier_arch']['input_dim'],
    hidden_dim=config_args['classifier_arch']['hidden_dim'],
    num_layers=config_args['classifier_arch']['num_layers'],
    unit_dropout=config_args['classifier_arch']['unit_dropout'],
    dropout_prob=config_args['classifier_arch']['dropout_prob']
)

# initialize main task classifier ...
nn_aux_classifier = FeedforwardClassifier(
    num_classes= config_args['aux_classifier_arch']['num_classes'],
    input_dim=config_args['aux_classifier_arch']['input_dim'],
    hidden_dim=config_args['aux_classifier_arch']['hidden_dim'],
    num_layers=config_args['aux_classifier_arch']['num_layers'],
    unit_dropout=config_args['aux_classifier_arch']['unit_dropout'],
    dropout_prob=config_args['aux_classifier_arch']['dropout_prob']
)


# initialize end-2-end adaptive LID classifier ...
adaptive_LID_classifier = AdaptiveSpeechClassifierI(
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier,
    adversarial_classifier=nn_aux_classifier
)

print('\nEnd-to-end LID classifier was initialized ...\n',
    adaptive_LID_classifier)


# define classification loss and auxiliary loss
cls_loss = nn.CrossEntropyLoss()
aux_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(adaptive_LID_classifier.parameters(), \
    lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = train_utils.make_train_state(config_args)

# this line was added due to RunTimeError
adaptive_LID_classifier.cuda()


num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']

# keep val acc for both src and tgt in this dict
balanced_acc_scores = collections.defaultdict(list)

print('Training started ...')


try:
    # iterate over training epochs ...
    for epoch_index in range(num_epochs):
        ### TRAINING ...
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset, set loss and acc to 0 for
        # both src & tgt, set train mode on, generate batch
        run_cls_loss, run_aux_loss = 0.0, 0.0
        run_cls_acc,  run_aux_acc  = 0.0, 0.0

        adaptive_LID_classifier.train()

        src_speech_dataset.set_mode('TRA')
        src_batch_generator = generate_batches(src_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        tgt_speech_dataset.set_mode('TRA')
        tgt_batch_generator = generate_batches(tgt_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        src_num_batches = src_speech_dataset.get_num_batches(batch_size)
        tgt_num_batches = tgt_speech_dataset.get_num_batches(batch_size)

        num_batches = min(src_num_batches, tgt_num_batches)

        # iterate over training batches
        dataset = zip(src_batch_generator, tgt_batch_generator)
        for batch_index, (src_batch_dict, tgt_batch_dict) in enumerate(dataset):
            # zero the gradients
            optimizer.zero_grad()

            # compute adaptation hyperparameter beta
            n = float(batch_index + epoch_index * num_batches)
            d =  (num_epochs * num_batches)
            p = (n/d)
            beta = 2.0 / (1. + np.exp(-10 * p)) - 1


            # forward pass and compute loss on source domain
            # generate source domain labels
            src_aux_tar = torch.zeros(batch_size, dtype=torch.long,
                device=config_args['device'])

            src_cls_tar = src_batch_dict['y_target']

            # forward pass
            src_cls_hat, src_aux_hat = adaptive_LID_classifier(
                x_in=src_batch_dict['x_data'], grl_lambda=beta
            )

            src_cls_loss = cls_loss(src_cls_hat, src_cls_tar)
            src_aux_loss = aux_loss(src_aux_hat, src_aux_tar)

            # forward pass and compute aux loss on target domain
            # generate source domain labels
            tgt_aux_tar = torch.ones(batch_size, dtype=torch.long,
                device=config_args['device'])

            _, tgt_aux_hat = adaptive_LID_classifier(
                x_in=tgt_batch_dict['x_data'], grl_lambda=beta
            )

            tgt_aux_loss = aux_loss(tgt_aux_hat, tgt_aux_tar)

            # add all losses
            loss = src_cls_loss + src_aux_loss + tgt_aux_loss

            # use loss to produce gradients
            loss.backward()

            # use optimizer to take gradient step
            optimizer.step()

            # compute different cls & aux losses
            batch_cls_loss = src_cls_loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

            batch_aux_loss = src_aux_loss.item() + tgt_aux_loss.item()
            run_aux_loss += (batch_aux_loss - run_aux_loss)/(batch_index + 1)

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            # compute running aux prediction acc. (domain prediction)
            src_aux_acc = train_utils.compute_accuracy(src_aux_hat, src_aux_tar)
            run_src_aux_acc = src_aux_acc /(batch_index + 1)
            tgt_aux_acc = train_utils.compute_accuracy(tgt_aux_hat, tgt_aux_tar)
            run_tgt_aux_acc = tgt_aux_acc /(batch_index + 1)

            run_aux_acc += ((run_src_aux_acc + run_tgt_aux_acc)/2) - \
                run_aux_acc / (batch_index + 1)


            # print summary
            print(f"{config_args['model_str']}    "
                f"TRA epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches}]    "
                f"cls-loss: {run_cls_loss:1.4f} :: "
                f"cls-acc: {run_cls_acc:2.2f}    "
                f"aux-loss: {run_aux_loss:1.4f} :: "
                f"aux-acc: {run_aux_acc:2.2f}"
            )


        # one epoch training is DONE! Update training state
        train_state['train_loss'].append(run_cls_loss)
        train_state['train_acc'].append(run_cls_acc)

        ### VALIDATION ...
        # run one validation pass over the validation split
        adaptive_LID_classifier.eval()

        src_speech_dataset.set_mode('DEV')
        src_batch_generator = generate_batches(src_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        tgt_speech_dataset.set_mode('DEV')
        tgt_batch_generator = generate_batches(tgt_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        src_num_batches = src_speech_dataset.get_num_batches(batch_size)
        tgt_num_batches = tgt_speech_dataset.get_num_batches(batch_size)

        num_batches = min(src_num_batches, tgt_num_batches)


        # iterate over validation batches
        # list to maintain model predictions on val set
        y_src_tar, y_src_hat = [], []
        y_tgt_tar, y_tgt_hat = [], []

        dataset = zip(src_batch_generator, tgt_batch_generator)
        for batch_index, (src_batch_dict, tgt_batch_dict) in enumerate(dataset):
            # forward pass and compute loss on source domain
            # generate source domain labels
            src_aux_tar = torch.zeros(batch_size, dtype=torch.long,
                device=config_args['device'])

            src_cls_tar = src_batch_dict['y_target']

            # forward pass
            src_cls_hat, src_aux_hat = adaptive_LID_classifier(
                x_in=src_batch_dict['x_data'], grl_lambda=beta
            )

            src_cls_loss = cls_loss(src_cls_hat, src_cls_tar)
            src_aux_loss = aux_loss(src_aux_hat, src_aux_tar)

            # forward pass and compute aux loss on target domain
            # generate source domain labels
            tgt_aux_tar = torch.ones(batch_size, dtype=torch.long,
                device=config_args['device'])

            tgt_cls_tar = tgt_batch_dict['y_target']

            tgt_cls_hat, tgt_aux_hat = adaptive_LID_classifier(
                x_in=tgt_batch_dict['x_data'], grl_lambda=beta
            )

            tgt_aux_loss = aux_loss(tgt_aux_hat, tgt_aux_tar)

            # compute different cls & aux losses
            batch_cls_loss = src_cls_loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

            batch_aux_loss = src_aux_loss.item() + tgt_aux_loss.item()
            run_aux_loss += (batch_aux_loss - run_aux_loss)/(batch_index + 1)

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            # compute running aux prediction acc. (domain prediction)
            src_aux_acc = train_utils.compute_accuracy(src_aux_hat, src_aux_tar)
            run_src_aux_acc = src_aux_acc /(batch_index + 1)
            tgt_aux_acc = train_utils.compute_accuracy(tgt_aux_hat, tgt_aux_tar)
            run_tgt_aux_acc = tgt_aux_acc /(batch_index + 1)

            run_aux_acc += ((run_src_aux_acc + run_tgt_aux_acc)/2) - \
                run_aux_acc / (batch_index + 1)


            # print summary
            print(f"{config_args['model_str']}    "
                f"VAL epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches}]    "
                f"cls-loss: {run_cls_loss:1.4f} :: "
                f"cls-acc: {run_cls_acc:2.2f}    "
                f"aux-loss: {run_aux_loss:1.4f} :: "
                f"aux-acc: {run_aux_acc:2.2f}"
            )

            # compute balanced acc calc
            y_src_tar, y_src_hat = train_utils.get_predictions_and_trues(
                src_cls_hat, src_cls_tar)

            y_tgt_tar, y_tgt_hat = train_utils.get_predictions_and_trues(
                tgt_cls_hat, tgt_cls_tar)

            y_src_tar.extend(y_src_tar); y_src_hat.extend(y_src_hat)
            y_tgt_tar.extend(y_tgt_tar); y_tgt_hat.extend(y_tgt_hat)


        # TRAIN & VAL iterations for one epoch is over ...
        train_state['val_loss'].append(run_cls_loss)
        train_state['val_acc'].append(run_cls_acc)

        # compute val performance on this epoch
        src_cls_acc_ep = balanced_accuracy_score(y_src_tar, y_src_hat)*100
        tgt_cls_acc_ep = balanced_accuracy_score(y_tgt_tar, y_tgt_hat)*100

        # update data strucutre for val perforamce metric
        balanced_acc_scores['src'].append(src_cls_acc_ep)
        balanced_acc_scores['tgt'].append(tgt_cls_acc_ep)


        train_state = train_utils.update_train_state(args=config_args,
            model=adaptive_LID_classifier,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])


        if train_state['stop_early']:
            break


except KeyboardInterrupt:
    print("Exiting loop")


# once training is over for the number of batches specified, check best epoch
for dataset in ['src', 'tgt']:
    acc_scores = balanced_acc_scores[dataset]
    for i, acc in enumerate(acc_scores):
        print("Validation Acc {} {:.3f}".format(i+1, acc))


    print('Best epoch by balanced acc: {:.3f} epoch {}'.format(max(acc_scores),
        1 + np.argmax(acc_scores)))
