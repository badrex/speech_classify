#! /usr/bin/env python3
# coding: utf-8

import os
import yaml
import sys

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
import torch.nn.functional as F
import torch.optim as optim

from nn_speech_models import *

# Training Routine
# Helper functions
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args['training_hyperparams']['learning_rate'],
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args['model_state_file']}


def update_train_state(args, model, train_state):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # save model
    torch.save(model.state_dict(), \
        train_state['model_filename'] + \
        str(train_state['epoch_index'] + 1) + '.pth')

    # save model after first epoch
    if train_state['epoch_index'] == 0:
        train_state['stop_early'] = False
        train_state['best_val_accuracy'] = train_state['val_acc'][-1]

    # after first epoch check early stopping criteria
    elif train_state['epoch_index'] >= 1:
        acc_t = train_state['val_acc'][-1]

        # if acc decreased, add one to early stopping criteria
        if acc_t <= train_state['best_val_accuracy']:
            # Update step
            train_state['early_stopping_step'] += 1

        else: # if acc improved
            train_state['best_val_accuracy'] = train_state['val_acc'][-1]

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        early_stop = train_state['early_stopping_step'] >= \
            args['training_hyperparams']['early_stopping_criteria']

        train_state['stop_early'] = early_stop

    return train_state


def compute_accuracy(y_pred, y_target):
    #y_target = y_target.cpu()
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def get_predictions(y_pred, y_target):
	"""Return indecies of predictions. """
	_, y_pred_indices = y_pred.max(dim=1)

	pred_labels = y_pred_indices.tolist()
	true_labels = y_target.tolist()

	return (true_labels, pred_labels)

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# obtain user input
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_pah = sys.argv[1] #'/LANG-ID-X/config_1.yml'

config_args = yaml.safe_load(open(config_file_pah))

config_args['model_id'] = '_'.join(str(ip) for ip in
    [
        config_args['model_arch']['nn_model'],
        config_args['model_arch']['bottleneck_size'],
        config_args['model_arch']['output_dim'],
        config_args['model_arch']['signal_dropout_prob'],
        config_args['input_signal_params']['feature_type'],
        config_args['input_signal_params']['signal_masking'],
        config_args['input_signal_params']['num_frames']
    ]
)

if config_args['expand_filepaths_to_save_dir']:
    config_args['model_state_file'] = os.path.join(
        config_args['model_save_dir'],
        '_'.join([config_args['model_state_file'],
        config_args['model_id']])
    )

    print("Expanded filepaths: ")
    print("\t{}".format(config_args['model_state_file']))


 # Check CUDA
if not torch.cuda.is_available():
    config_args['cuda'] = False

config_args['device'] = torch.device("cuda" if config_args['cuda'] else "cpu")

print("Using CUDA: {}".format(config_args['cuda']))

# Set seed for reproducibility
set_seed_everywhere(config_args['seed'], config_args['cuda'])

# handle dirs
handle_dirs(config_args['model_save_dir'])


##### HERE IT ALL STARTS ...
# vectorizer ...
speech_df = pd.read_csv(config_args['speech_metadata'],
    delimiter="\t", encoding='utf-8')

label_set=config_args['language_set'].split()

# make sure no utterances with 0 duration such as
speech_df = speech_df[(speech_df.duration!=0)]

speech_df = speech_df[(speech_df['language'].isin(label_set))]

##### HERE IT ALL STARTS ...
# source vectorizer ...
speech_df = pd.read_csv(config_args['speech_metadata'],
    delimiter="\t", encoding='utf-8')

label_set=config_args['language_set'].split()


n_samples = config_args['eval_samples']

# make sure no utterances with 3 duration such as
train_df = speech_df[(speech_df.duration>3.0) & (speech_df.split=='TRA')]

pos_df = train_df[((train_df.gender=='male') &
                   (train_df.language==config_args['pos_lang']))].sample(n=1000, random_state=1)

neg_df = train_df[((train_df.gender=='female') &
                   (train_df.language==config_args['neg_lang']))].sample(n=1000, random_state=1)

pos_val_df = speech_df[(speech_df.duration>3.0) &
                       (speech_df.split=='DEV') &
                       (speech_df.gender=='male') &
                       (speech_df.language==config_args['pos_lang'])].sample(n=n_samples, random_state=1)

neg_val_df = speech_df[(speech_df.duration>3.0) &
                       (speech_df.split=='DEV') &
                       (speech_df.gender=='female') &
                       (speech_df.language==config_args['neg_lang'])].sample(n=n_samples, random_state=1)

pos_eval_df = speech_df[(speech_df.duration>3.0) &
                       (speech_df.split=='EVA') &
                       #(speech_df.gender=='female') &
                       (speech_df.language==config_args['pos_lang'])].sample(n=n_samples, random_state=1)

neg_eval_df = speech_df[(speech_df.duration>3.0) &
                       (speech_df.split=='EVA') &
                       #(speech_df.gender=='male') &
                       (speech_df.language==config_args['neg_lang'])].sample(n=n_samples, random_state=1)

sess_speech_df = pd.concat([pos_df, neg_df, pos_val_df, neg_val_df, pos_eval_df, neg_eval_df])




print(sess_speech_df.columns.values.tolist())

print(sess_speech_df.head())

sess_speech_df.rename(columns={0:'uttr_id'}, inplace=True)

print(sess_speech_df.head())


speech_vectorizer = SpeechFeaturizer(
    data_dir=config_args['preprocessed_data_dir'],
    feature_type= config_args['input_signal_params']['feature_type'],
    label_set=config_args['language_set'].split(), # split str into list of str
    max_num_frames=config_args['input_signal_params']['max_num_frames'],
    num_frames=config_args['input_signal_params']['num_frames'],
    feature_dim=config_args['model_arch']['feature_dim'],
    start_index=config_args['input_signal_params']['start_index'],
    end_index=config_args['input_signal_params']['end_index']
)
print(speech_vectorizer.index2label)

# data loader ....
speech_dataset = LID_Dataset(sess_speech_df, speech_vectorizer)




if config_args['model_arch']['nn_model'] == 'ConvNet':
    nn_LID_model = ConvNet_LID(
        feature_dim=config_args['model_arch']['feature_dim'],
        bottleneck=config_args['model_arch']['bottleneck'],
        bottleneck_size=config_args['model_arch']['bottleneck_size'],
        output_dim=config_args['model_arch']['output_dim'],
        dropout_frames=config_args['model_arch']['frame_dropout'],
        dropout_features=config_args['model_arch']['feature_dropout'],
        signal_dropout_prob=config_args['model_arch']['signal_dropout_prob'],
        num_channels=config_args['model_arch']['num_channels'],
        num_classes= len(label_set),   # or config_args['model_arch']['num_classes'],
        filter_sizes=config_args['model_arch']['filter_sizes'],
        stride_steps=config_args['model_arch']['stride_steps'],
        pooling_type=config_args['model_arch']['pooling_type']
    )

elif config_args['model_arch']['nn_model'] == 'Linear':
    nn_LID_model = LinearLID(
    feature_dim=config_args['model_arch']['feature_dim'],
    num_classes= len(label_set)
    )

elif config_args['model_arch']['nn_model'] == 'MLP':
    nn_LID_model = MLPNetLID(
    feature_dim=config_args['model_arch']['feature_dim'],
    num_classes= len(label_set)
    )

elif config_args['model_arch']['nn_model'] == 'MLP2':
    nn_LID_model = MLPNetLID2(
    feature_dim=config_args['model_arch']['feature_dim'],
    num_classes= len(label_set)
    )

else:
    raise NotImplementedError

print(nn_LID_model)

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(nn_LID_model.parameters(), \
    lr=config_args['training_hyperparams']['learning_rate'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                mode='min', factor=0.5,
                patience=1)

train_state = make_train_state(config_args)

# this line was added due to RunTimeError
nn_LID_model.cuda()

val_balanced_acc_scores = []
eval_balanced_acc_scores = []

try:
    print('Training started.')
    for epoch_index in range(config_args['training_hyperparams']['num_epochs']):
        train_state['epoch_index'] = epoch_index

        # Iterate over training dataset

        # setup: batch generator, set loss and acc to 0, set train mode on
        speech_dataset.set_mode('TRA') # , debug_mode='shuffle
        total_batches = speech_dataset.get_num_batches(
            config_args['training_hyperparams']['batch_size'])

        batch_generator = generate_batches(speech_dataset,
            batch_size=config_args['training_hyperparams']['batch_size'],
            device=config_args['device'])

        running_loss = 0.0
        running_acc = 0.0

        nn_LID_model.train()


        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = nn_LID_model(x_in=batch_dict['x_data'], frame_shuffle=True)

            # step 3. compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            #
            print('{} \t Train epoch [{:>2}/{}][{:>4}/{}]\t loss: {:.6f}\t acc: {:.2f}'.format(
                config_args['model_id'],
                epoch_index + 1,
                config_args['training_hyperparams']['num_epochs'],
                batch_index + 1,
                total_batches,
                running_loss,
                running_acc)
            )


        train_state['train_loss'].append(running_loss)
        train_state['train_acc'].append(running_acc)

        # Iterate over val dataset
        # setup: batch generator, set loss and acc to 0; set eval mode on
        speech_dataset.set_mode('EVA')
        total_batches = speech_dataset.get_num_batches(
            config_args['training_hyperparams']['batch_size'])

        batch_generator = generate_batches(speech_dataset,
            batch_size=config_args['training_hyperparams']['batch_size'],
            device=config_args['device'])

        running_loss = 0.
        running_acc = 0.

        nn_LID_model.eval()

        y_true, y_pred = [], []

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            nn_y_pred = nn_LID_model(x_in=batch_dict['x_data'])

            # step 3. compute the loss
            loss = loss_func(nn_y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(nn_y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            #
            print('{}\t Eval epoch [{:>2}][{:>4}/{}]\t loss: {:.6f}\t acc: {:.2f}'.format(
                config_args['model_id'],
                epoch_index + 1,
                batch_index + 1,
                total_batches,
                running_loss,
                running_acc)
            )

            # balanced acc calc
            true_labels, pred_labels = get_predictions(nn_y_pred, batch_dict['y_target'])

            y_true.extend(true_labels)
            y_pred.extend(pred_labels)

        acc_score = balanced_accuracy_score(y_true, y_pred)*100

        eval_balanced_acc_scores.append(acc_score)

        ############## DEV

        speech_dataset.set_mode('DEV')
        total_batches = speech_dataset.get_num_batches(
            config_args['training_hyperparams']['batch_size'])

        batch_generator = generate_batches(speech_dataset,
            batch_size=config_args['training_hyperparams']['batch_size'],
            device=config_args['device'])

        running_loss = 0.
        running_acc = 0.

        nn_LID_model.eval()

        y_true, y_pred = [], []

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            nn_y_pred = nn_LID_model(x_in=batch_dict['x_data'], frame_shuffle=True)

            # step 3. compute the loss
            loss = loss_func(nn_y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(nn_y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            #
            print('{}\t  Val epoch [{:>2}][{:>4}/{}]\t loss: {:.6f}\t acc: {:.2f}'.format(
                config_args['model_id'],
                epoch_index + 1,
                batch_index + 1,
                total_batches,
                running_loss,
                running_acc)
            )

            # balanced acc calc
            true_labels, pred_labels = get_predictions(nn_y_pred, batch_dict['y_target'])

            y_true.extend(true_labels)
            y_pred.extend(pred_labels)


        acc_score = balanced_accuracy_score(y_true, y_pred)*100

        val_balanced_acc_scores.append(acc_score)

        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        train_state = update_train_state(args=config_args,
            model=nn_LID_model,
            train_state=train_state
        )

        scheduler.step(train_state['val_loss'][-1])

        #print('\nEpoch {:>2} is over, '.format(epoch_index + 1), end='')
        #print('val loss: {:.6f}, val acc: {:.2f}'.format(train_state['val_loss'][-1], train_state['val_acc'][-1]))
        #print('----------------------')

        if train_state['stop_early']:
            break


except KeyboardInterrupt:
    print("Exiting loop")

for i, acc in enumerate(val_balanced_acc_scores):
    print("Validation Acc {} {:.3f}".format(i+1, acc))


print('Best epoch by balanced acc: {:.3f} epoch {}'.format(max(val_balanced_acc_scores),
    1 + np.argmax(val_balanced_acc_scores)))


for i, acc in enumerate(eval_balanced_acc_scores):
    print("Eval Acc {} {:.3f}".format(i+1, acc))


print('Best epoch by balanced acc: {:.3f} epoch {}'.format(max(eval_balanced_acc_scores),
    1 + np.argmax(eval_balanced_acc_scores)))
