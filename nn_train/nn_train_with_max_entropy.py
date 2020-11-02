#! /usr/bin/env python3
# coding: utf-8

# get parent directory to be visible
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import yaml
import pprint
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
from nn_loss_functions import EntropyLoss
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


#  data loader ...
src_speech_dataset = SpeechDataset(src_speech_df, src_speech_featurizer)


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


# initialize end-2-end LID classifier ...
LID_classifier = SpeechClassifier(
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
)

print('\nEnd-to-end LID classifier was initialized ...\n',
    LID_classifier)


# define classification loss
cls_loss = nn.CrossEntropyLoss()
entropy_loss = EntropyLoss() #reduction='mean'


optimizer = optim.Adam(LID_classifier.parameters(), \
    lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = train_utils.make_train_state(config_args)

# this line was added due to RunTimeError
LID_classifier.cuda()


num_epochs = config_args['training_hyperparams']['num_epochs']
batch_size = config_args['training_hyperparams']['batch_size']

# keep val acc for both src and tgt in this dict
balanced_acc_scores = collections.defaultdict(list)

print('Training started ...')
torch.autograd.set_detect_anomaly(True)


try:
    # iterate over training epochs ...
    for epoch_index in range(num_epochs):
        ### TRAINING ...
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset, set loss and acc to 0
        # set train mode on, generate batch
        run_cls_loss, run_cls_acc = 0.0, 0.0
        run_ent_loss = 0.0

        LID_classifier.train()

        src_speech_dataset.set_mode('TRA')
        src_batch_generator = generate_batches(src_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )


        num_batches = src_speech_dataset.get_num_batches(batch_size)

        # iterate over training batches
        for batch_index, src_batch_dict in enumerate(src_batch_generator):
            # zero the gradients
            optimizer.zero_grad()

			# compute adaptation hyperparameter beta
            n = float(batch_index + epoch_index * num_batches)
            d =  (num_epochs * num_batches)
            p = (n/d)
            beta = 2.0 / (1. + np.exp(-10 * p)) - 1

            # forward pass and compute loss on source domain
            src_cls_tar = src_batch_dict['y_target'][:int(batch_size/2)]

            #print(src_cls_tar.shape)

            #if batch_index % 2 == 0:
            # normal forward pass to compute classification entropy
            src_cls_hat = LID_classifier(x_in=src_batch_dict['x_data'][:int(batch_size/2)])
            c_loss = cls_loss(src_cls_hat, src_cls_tar)

            #else:
			# compute entropy loss on shuffled sequences
            src_cls_hat_sh = LID_classifier(x_in=src_batch_dict['x_data'][int(batch_size/2):], shuffle_frames=True)
            e_loss =  entropy_loss(src_cls_hat_sh).mean()

            # use loss to produce gradients
            loss = c_loss + (- beta * e_loss)
            loss.backward()#; e_loss.backward()
            #print('hhhh')
            # use optimizer to take gradient step
            optimizer.step()

            # compute different cls & ent losses
            batch_cls_loss = loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

			#  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            #  compute running ent loss
            #batch_ent_loss = e_loss.item()
            #run_ent_loss += (batch_ent_loss - run_ent_loss)/(batch_index + 1)

            # print summary
            print(f"{config_args['model_str']}    "
                f"TRA epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches:>4}]    "
                f"cls - loss: {batch_cls_loss:1.4f} :: "
                f"acc: {run_cls_acc:2.2f}     "
				f"beta: {beta:1.4f} :: "
            )


        # one epoch training is DONE! Update training state
        train_state['train_loss'].append(run_cls_loss)
        train_state['train_acc'].append(run_cls_acc)

        ### VALIDATION ...
        # run one validation pass over the validation split
        LID_classifier.eval()

        src_speech_dataset.set_mode('DEV')
        src_batch_generator = generate_batches(src_speech_dataset,
            batch_size=batch_size, device=config_args['device']
        )

        num_batches = src_speech_dataset.get_num_batches(batch_size)

        # iterate over validation batches
        # list to maintain model predictions on val set
        y_src_tar, y_src_hat = [], []


        for batch_index, src_batch_dict in enumerate(src_batch_generator):
            # forward pass and compute loss on source domain

            src_cls_tar = src_batch_dict['y_target']

            # forward pass
            src_cls_hat = LID_classifier(x_in=src_batch_dict['x_data'])

            src_cls_loss = cls_loss(src_cls_hat, src_cls_tar)

            # compute different cls & aux losses
            batch_cls_loss = src_cls_loss.item()
            run_cls_loss += (batch_cls_loss - run_cls_loss)/(batch_index + 1)

            #  compute running source cls accuracy
            src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
            run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

            # print summary
            print(f"{config_args['model_str']}    "
                f"VAL epoch [{epoch_index + 1:>2}/{num_epochs}]"
                f"[{batch_index + 1:>4}/{num_batches:>4}]    "
                f"cls - loss: {run_cls_loss:1.4f} :: "
                f"acc: {run_cls_acc:2.2f}"
            )

			# get predictions
            batch_y_src_hat, batch_y_src_tar = train_utils.get_predictions_and_trues(
				src_cls_hat, src_cls_tar
			)
            y_src_tar.extend(batch_y_src_tar); y_src_hat.extend(batch_y_src_hat)

        # TRAIN & VAL iterations for one epoch is over ...
        train_state['val_loss'].append(run_cls_loss)
        train_state['val_acc'].append(run_cls_acc)

        # compute val performance on this epoch using balanced acc
        src_cls_acc_ep = balanced_accuracy_score(y_src_tar, y_src_hat)*100

        # update data strucutre for val perforamce metric
        balanced_acc_scores['src'].append(src_cls_acc_ep)

        train_state = train_utils.update_train_state(args=config_args,
            model=LID_classifier,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])


        if train_state['stop_early']:
            break

except KeyboardInterrupt:
    print("Exiting loop")


# once training is over for the number of batches specified, check best epoch
for dataset in ['src']:
    acc_scores = balanced_acc_scores[dataset]
    for i, acc in enumerate(acc_scores):
        print("Validation Acc {} {:.3f}".format(i+1, acc))


    print('Best {} model by balanced acc: {:.3f} epoch {}'.format(
		config_args['model_str'],
		max(acc_scores),
        1 + np.argmax(acc_scores))
	)
