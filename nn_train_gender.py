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
# featurizer ...
speech_df = pd.read_csv(config_args['speech_metadata'],
    delimiter="\t", encoding='utf-8')

label_set=config_args['language_set'].split()

# sample only target labels and more than 3.0 seconds
speech_df = speech_df[(speech_df.language.isin(label_set)) &
    (speech_df.duration>3.0) ]

pos_label, neg_label = label_set

num_train_samples = config_args['train_samples']
num_eval_samples = config_args['eval_samples']

# make and sample dataset splits for the experiment ...
experiment_df_list = []

# make splits for training and matched eval set
for _split in ['TRA', 'DEV']:

    num_samples = num_train_samples if _split == 'TRA' else num_eval_samples

    pos_df = speech_df[((speech_df.gender=='male') &
        (speech_df.split==_split) &
        (speech_df.language==pos_label))].sample(
        n=num_samples, random_state=1
    )

    neg_df = speech_df[((speech_df.gender=='female') &
        (speech_df.split==_split) &
        (speech_df.language==neg_label))].sample(
        n=num_samples, random_state=1
    )

    experiment_df_list.extend([pos_df, neg_df])

# mismtached eval set
pos_eval_df = speech_df[(speech_df.split=='EVA') &
   (speech_df.gender=='female') &
   (speech_df.language==pos_label)].sample(n=num_eval_samples, random_state=1)

neg_eval_df = speech_df[(speech_df.split=='EVA') &
   (speech_df.gender=='male') &
   (speech_df.language==neg_label)].sample(n=num_eval_samples, random_state=1)


experiment_df_list.extend([pos_eval_df, neg_eval_df])

# make a single dataframe
sess_speech_df = pd.concat(experiment_df_list)


#print(sess_speech_df.columns.values.tolist())
#print(sess_speech_df.head())

sess_speech_df.rename(columns={0:'uttr_id'}, inplace=True)


speech_featurizer = SpeechFeaturizer(
    data_dir=config_args['speech_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['language_set'].split(), # split str into list of str
    num_frames=config_args['input_signal_params']['num_frames'],
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    spectral_dim=config_args['encoder_arch']['spectral_dim'],
    start_index=config_args['input_signal_params']['start_index'],
    end_index=config_args['input_signal_params']['end_index']
)

print('SpeechFeaturizer was initialized: ', speech_featurizer.index2label)

# data loader ...
speech_dataset = SpeechDataset(sess_speech_df, speech_featurizer)

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

#print('\nConv Speech Encoder was initialized ...\n', nn_speech_encoder)



# initialize task classifier ...
nn_task_classifier = FeedforwardClassifier(
    num_classes= config_args['classifier_arch']['num_classes'], # or len(label_set)
    input_dim=config_args['classifier_arch']['input_dim'],
    hidden_dim=config_args['classifier_arch']['hidden_dim'],
    num_layers=config_args['classifier_arch']['num_layers'],
    unit_dropout=config_args['classifier_arch']['unit_dropout'],
    dropout_prob=config_args['classifier_arch']['dropout_prob']
)

#print('\nTask classifier was initialized ...\n', nn_task_classifier)

# initialize end-2-end LID classifier ...
LID_classifier = SpeechClassifier(
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
)

print('\nEnd-to-end LID classifier was initialized ...\n', LID_classifier)

# if config_args['encoder_arch']['encoder_model'] == 'ConvNet':
#     nn_LID_model = ConvNet_LID(
#         spectral_dim=config_args['encoder_arch']['spectral_dim'],
#         bottleneck=config_args['encoder_arch']['bottleneck'],
#         bottleneck_size=config_args['encoder_arch']['bottleneck_size'],
#         output_dim=config_args['encoder_arch']['output_dim'],
#         dropout_frames=config_args['encoder_arch']['frame_dropout'],
#         dropout_spectral_features=config_args['encoder_arch']['feature_dropout'],
#         signal_dropout_prob=config_args['encoder_arch']['signal_dropout_prob'],
#         num_channels=config_args['encoder_arch']['num_channels'],
#         num_classes= len(label_set),   # or config_args['encoder_arch']['num_classes'],
#         filter_sizes=config_args['encoder_arch']['filter_sizes'],
#         stride_steps=config_args['encoder_arch']['stride_steps'],
#         pooling_type=config_args['encoder_arch']['pooling_type']
#     )
#
# elif config_args['encoder_arch']['encoder_model'] == 'Linear':
#     nn_LID_model = LinearLID(
#     spectral_dim=config_args['encoder_arch']['spectral_dim'],
#     num_classes= len(label_set)
#     )
#
# elif config_args['encoder_arch']['encoder_model'] == 'MLP':
#     nn_LID_model = MLPNetLID(
#     spectral_dim=config_args['encoder_arch']['spectral_dim'],
#     num_classes= len(label_set)
#     )
#
# elif config_args['encoder_arch']['encoder_model'] == 'MLP2':
#     nn_LID_model = MLPNetLID2(
#     spectral_dim=config_args['encoder_arch']['spectral_dim'],
#     num_classes= len(label_set)
#     )
#
# else:
#     raise NotImplementedError


loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(LID_classifier.parameters(), \
    lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = train_utils.make_train_state(config_args)

# this line was added due to RunTimeError
LID_classifier.cuda()

balanced_acc_scores = collections.defaultdict(list)


try:
    print('Training started.')
    for epoch_index in range(config_args['training_hyperparams']['num_epochs']):
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: set loss and acc to 0, set train mode on, generate batch
        running_loss = 0.0
        running_acc = 0.0
        speech_dataset.set_mode('TRA')
        LID_classifier.train()


        total_num_batches = speech_dataset.get_num_batches(
            config_args['training_hyperparams']['batch_size']
        )

        batch_generator = generate_batches(speech_dataset,
            batch_size=config_args['training_hyperparams']['batch_size'],
            device=config_args['device']
        )

        for batch_index, batch_dict in enumerate(batch_generator):

            # zero the gradients
            optimizer.zero_grad()

            # forward pass through net
            y_hat = LID_classifier(x_in=batch_dict['x_data'], shuffle_frames=False) # shuffle_frames
            y_tar = batch_dict['y_target']

            # compute the loss between predicted label and target label
            loss = loss_func(y_hat, y_tar)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # loss to produce gradients and backprop
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = train_utils.compute_accuracy(y_hat, y_tar)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            print(f"{config_args['model_str']}    "
                f"TRA epoch [{epoch_index + 1:>2}"
                f"/{config_args['training_hyperparams']['num_epochs']}]"
                f"[{batch_index + 1:>4}/{total_num_batches}]    "
                f"loss: {running_loss:.4f}    "
                f"acc: {running_acc:.2f}"
            )



        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over evaluation dataset: DEV and Eval
        for _split in ['DEV', 'EVA']:
            # set split
            speech_dataset.set_mode(_split)

            total_num_batches = speech_dataset.get_num_batches(
                config_args['training_hyperparams']['batch_size'])

            batch_generator = generate_batches(speech_dataset,
                batch_size=config_args['training_hyperparams']['batch_size'],
                device=config_args['device'])

            running_loss = 0.
            running_acc = 0.

            LID_classifier.eval()

            y_hat_list, y_tar_list = [], []

            for batch_index, batch_dict in enumerate(batch_generator):

                y_hat = LID_classifier(x_in=batch_dict['x_data'])
                y_tar = batch_dict['y_target']

                loss = loss_func(y_hat, y_tar)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                acc_t = train_utils.compute_accuracy(y_hat, y_tar)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # get labels and compute balanced acc.
                y_hat_batch, y_tar_batch = train_utils.get_predictions_and_trues(
                    y_hat, y_tar)

                y_hat_list.extend(y_hat_batch)
                y_tar_list.extend(y_tar_batch)

                print(f"{config_args['model_str']}    "
                    f"{_split} epoch [{epoch_index + 1:>2}"
                    f"/{config_args['training_hyperparams']['num_epochs']}]"
                    f"[{batch_index + 1:>4}/{total_num_batches:>2}]    "
                    f"loss: {running_loss:.4f}    "
                    f"acc: {running_acc:.2f}"
                )


            acc_score = balanced_accuracy_score(y_hat_list, y_tar_list)*100
            balanced_acc_scores[_split].append(acc_score)

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = train_utils.update_train_state(args=config_args,
            model=LID_classifier,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])


        if train_state['stop_early']:
            break


except KeyboardInterrupt:
    print("Exiting loop")


for _split in ['DEV', 'EVA']:

    print()
    for i, acc in enumerate(balanced_acc_scores[_split]):
        print(f"{_split} Acc {i+1} {acc:.3f}")

    print(f"Best epoch by balanced acc: {max(balanced_acc_scores[_split]):.3f} "
        f"epoch {1 + np.argmax(balanced_acc_scores[_split])}"
    )
