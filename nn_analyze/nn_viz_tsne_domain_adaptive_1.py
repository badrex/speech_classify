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

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torch.nn as nn
import torch.optim as optim


# for plots
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.text import TextPath

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
        config_args['encoder_arch']['encoder_model'],
        config_args['classifier_arch']['input_dim'],
        config_args['classifier_arch']['hidden_dim'],
        config_args['input_signal_params']['feature_type']
    ]
)

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

src_label_set=config_args['language_set'].split()



# get only target labels and more than 3.0 seconds
src_speech_df = src_speech_df[
    (src_speech_df.language.isin(src_label_set)) &
    (src_speech_df.duration>3.0) &
	(src_speech_df.split=='TRA')
].sample(n=config_args['num_samples'], random_state=1)

src_speech_featurizer = SpeechFeaturizer(
    data_dir=config_args['source_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['language_set'].split(), # split str into list of str
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

tgt_label_set=config_args['language_set'].split()


# get only target labels and more than 3.0 seconds
tgt_speech_df = tgt_speech_df[
    (tgt_speech_df.language.isin(tgt_label_set)) &
    (tgt_speech_df.duration>3.0) &
	(tgt_speech_df.split=='TRA')
].sample(n=config_args['num_samples'], random_state=1)

tgt_speech_featurizer = SpeechFeaturizer(
    data_dir=config_args['target_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['language_set'].split(), # split str into list of str
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

state_dict = torch.load(config_args['model_save_dir'] + config_args['pretrained_model'])
adaptive_LID_classifier.load_state_dict(state_dict)


# this line was added due to RunTimeError
adaptive_LID_classifier.cuda()


batch_size = config_args['batch_size']


try:
	### EVAL ...
	adaptive_LID_classifier.eval()

	src_speech_dataset.set_mode('TRA')

	src_batch_generator = generate_batches(src_speech_dataset,
	    batch_size=batch_size,
		device=config_args['device'],
		drop_last_batch=False
	)

	num_batches = src_speech_dataset.get_num_batches(batch_size)

	# iterate over sample batches
	# list to maintain model representations on source
	src_vectors = []
	tgt_vectors = []

	src_label = []
	tgt_label = []

	for batch_index, src_batch_dict in enumerate(src_batch_generator):
		# forward pass and compute loss on source domain

		src_cls_tar = [src_speech_featurizer.index2label[i] \
			for i in src_batch_dict['y_target'].cpu().tolist()]

		src_label.extend(src_cls_tar)

		# forward pass
		vectors = adaptive_LID_classifier(x_in=src_batch_dict['x_data'],
			return_vector=True)

		# print summary
		print(f"{config_args['pretrained_model']}    "
		    f"[{batch_index + 1:>4}/{num_batches:>4}]"
		)

		src_vectors.extend([v.detach().cpu().numpy() for v in vectors])


	tgt_batch_generator = generate_batches(tgt_speech_dataset,
	    batch_size=batch_size,
		device=config_args['device'],
		drop_last_batch=False
	)

	num_batches = tgt_speech_dataset.get_num_batches(batch_size)

	for batch_index, tgt_batch_dict in enumerate(tgt_batch_generator):
		# forward pass and compute loss on source domain

		tgt_cls_tar = [tgt_speech_featurizer.index2label[i] \
			for i in tgt_batch_dict['y_target'].cpu().tolist()]

		tgt_label.extend(tgt_cls_tar)

		# forward pass
		vectors = adaptive_LID_classifier(x_in=tgt_batch_dict['x_data'],
			return_vector=True)

		# print summary
		print(f"{config_args['pretrained_model']}    "
		    f"[{batch_index + 1:>4}/{num_batches:>4}]"
		)

		tgt_vectors.extend([v.detach().cpu().numpy() for v in vectors])

	#print(len(src_vectors), len(tgt_vectors))

except KeyboardInterrupt:
    print("Exiting loop")


# t-SNE viz. start here

 # make one numpy array
d_labels = ['src' for _ in range(config_args['num_samples']) ] + \
	['tgt' for _ in range(config_args['num_samples'])]

y_labels = src_label + tgt_label

x_vectors = src_vectors + tgt_vectors

# shuffle data points to make plots look better
indices = [i for i in range(len(y_labels))]
sh_idx =  indices[:]

import random
random.shuffle(sh_idx)

idx2idx = {i: j for i, j in zip(sh_idx, indices)}

x_vectors = [x_vectors[i] for i in sh_idx]
d_labels = [d_labels[i] for i in sh_idx]
y_labels = [y_labels[i] for i in sh_idx]


# t-SNE
print('t-SNE computation ...')
tsne = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=2000)
X_2d = tsne.fit_transform(np.array(x_vectors))




# plotting
print('Plotting ...')
plt.figure(figsize=(10, 10))

d2c = {
    'src': 'green',
    'tgt': 'red'
}


for (x_1, x_2), d in zip(X_2d, d_labels):
    plt.plot(x_1, x_2, c=d2c[d], marker='o', markersize=8, alpha=0.5)

plt.xticks([])
plt.yticks([])
plt.box(False)
# _path = 'plots_and_figures/' + model_id + '.' + data_split.lower() + '_3_50' + '.pdf'

plt.savefig(config_args['to_save_file'] + '_domain.pdf' )
plt.show()


# plotting
print('Plotting ...')
plt.figure(figsize=(10, 10))

l2c = {
    'BUL': 'purple',
    'HRV': 'deepskyblue',
    'CZE': 'grey',
    'POL': 'royalblue',
    'RUS': 'orangered',
    'UKR': 'gold'
}

d2m = {
    'GP': "v",
    'RB': "^"
}

for (x_1, x_2), l in zip(X_2d, y_labels):
    plt.plot(x_1, x_2, c=l2c[l], marker='o',  markersize=8, alpha=0.5) # markeredgecolor=(0, 0, 0, 1),
    #plt.annotate(uttr_id, xy=(x_1, x_2), xytext=(0, 0), fontsize=1,textcoords='offset points', alpha=1)
    #plt.plot(x_1, x_2, c='k', marker=path, markersize=10, alpha=0.75)

plt.xticks([])
plt.yticks([])
plt.box(False)
plt.savefig(config_args['to_save_file'] + '_label.pdf' )
plt.show()
