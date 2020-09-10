# Helper functions to train neural models

import os
import torch
import numpy as np


def make_train_state(args):
    return {
        'stop_early': False,
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
        'model_filename': args['model_state_file']
    }


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
    torch.save(model.state_dict(), train_state['model_filename'] + '_' + \
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


def get_predictions_and_trues(y_pred, y_target):
	"""Return indecies of predictions. """
	_, y_pred_indices = y_pred.max(dim=1)

	pred_labels = y_pred_indices.tolist()
	true_labels = y_target.tolist()

	return (pred_labels, true_labels)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
