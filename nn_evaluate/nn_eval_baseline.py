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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, \
	precision_recall_fscore_support

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
		max_num_frames=config_args['input_signal_params']['max_num_frames'],
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
baseline_LID_classifier = SpeechClassifier(
    speech_segment_encoder=nn_speech_encoder,
    task_classifier=nn_task_classifier
)

print('\nEnd-to-end LID classifier was initialized ...\n',
    baseline_LID_classifier)


state_dict = torch.load(config_args['model_save_dir'] + config_args['pretrained_model'])
baseline_LID_classifier.load_state_dict(state_dict)

# this line was added due to RunTimeError
baseline_LID_classifier.cuda()


batch_size = config_args['batch_size']


try:
	### VALIDATION ...
	# run one validation pass over the validation split
	baseline_LID_classifier.eval()

	src_speech_dataset.set_mode(config_args['eval_split'])

	src_batch_generator = generate_batches(src_speech_dataset,
	    batch_size=batch_size,
		device=config_args['device'],
		drop_last_batch=False
	)

	num_batches = src_speech_dataset.get_num_batches(batch_size)

	# iterate over validation batches
	# list to maintain model predictions on val set
	y_src_tar, y_src_hat = [], []

	run_cls_acc = 0


	for batch_index, src_batch_dict in enumerate(src_batch_generator):
		# forward pass and compute loss on source domain

		src_cls_tar = src_batch_dict['y_target']

		# forward pass
		src_cls_hat = baseline_LID_classifier(x_in=src_batch_dict['x_data'])


		#  compute running source cls accuracy
		src_cls_acc = train_utils.compute_accuracy(src_cls_hat, src_cls_tar)
		run_cls_acc += (src_cls_acc - run_cls_acc)/(batch_index + 1)

		# print summary
		print(f"{config_args['pretrained_model']}    "
		    f"[{batch_index + 1:>4}/{num_batches:>4}]    "
		    f"acc: {run_cls_acc:2.2f}"
		)

		# compute balanced acc calc
		batch_y_src_hat, batch_y_src_tar = train_utils.get_predictions_and_trues(
		    src_cls_hat, src_cls_tar)

		y_src_tar.extend(batch_y_src_tar); y_src_hat.extend(batch_y_src_hat)


	# compute val performance on this epoch
	#print(y_src_tar, y_src_hat)
	print(collections.Counter(y_src_tar))
	print(collections.Counter(y_src_hat))
	src_cls_acc = balanced_accuracy_score(y_src_tar, y_src_hat)*100
	print(f"{src_cls_acc:2.2f}")

	src_cls_acc = accuracy_score(y_src_tar, y_src_hat)*100
	print(f"{src_cls_acc:2.2f}")

	task_labels = list(src_speech_featurizer.index2label.keys())

	P, R, F, _ = precision_recall_fscore_support(y_src_tar, y_src_hat, average=None, labels=task_labels)

	print("{:>15} {:>10} {:>10} {:>10}".format('Language', 'P', 'R', 'F'))

	for i, c in enumerate(task_labels):
		lang = src_speech_featurizer.index2label[c]
		print("{:>15} {:10.2f} {:10.2f} {:10.2f}".format(lang, P[i]*100, R[i]*100, F[i]*100))

	mP, mR, mF, _ = precision_recall_fscore_support(y_src_tar, y_src_hat, average='macro')
	print("{:>15} {:10.2f} {:10.2f} {:10.2f}".format('Macro Avg', mP*100, mR*100, mF*100))

	uP, uR, uF, _ = precision_recall_fscore_support(y_src_tar, y_src_hat, average='micro')
	print("{:>15} {:10.2f} {:10.2f} {:10.2f}".format('Micro Avg', uP*100, uR*100, uF*100))

except KeyboardInterrupt:
    print("Exiting loop")
