# coding: utf-8
# edited from remote client! :D

import math
import numpy as np
import random

import torch
from torch import Tensor
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Function


##### CLASS SpeechFeaturizer: A featurizer for speech classification problems
class SpeechFeaturizer(object):
    """ The Featurizer handles the low-level speech features and labels. """

    def __init__(self,
        data_dir,
        feature_type,
        label_set,
        num_frames,
        max_num_frames,
        spectral_dim=13,
        start_index=0,
        end_index=13
    ):
        """
        Args:
            data_dir (str): the path to the data on disk to read .npy files
            features_type (str): low-level speech features, e.g., MFCCs
            label_set (set): the set of labels (e.g., 'RUS', 'CZE', etc.)
            num_frames (int): the number of acoustic frames to sample from the
                speech signal, e.g., 300 frames is equivalent to 3 seconds
            max_num_frames (int): the max number of acoustic frames in input
                the diff. (max_num_frames - num_frames) is padded with zeros
            spectral_dim (int): the num of spectral components (default 13)
            start_index (int): the index of the 1st component (default: 0)
            end_index (int): the index of the last component (default: 13)
        """
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.spectral_dim = spectral_dim
        self.start_index = start_index
        self.end_index = end_index

        # get set of the labels in the dataset
        self.label_set = label_set

        # obtain index --> label dict
        self.index2label = {idx:lbl for (idx, lbl) in enumerate(self.label_set)}


        # obtain label --> index dict
        self.label2index = {lbl:idx for (idx, lbl) in self.index2label.items()}



    def transform_input_X(self,
        uttr_id,
        num_frames=None,
        max_num_frames=None,
        spectral_dim=None,
        start_index=None,
        end_index=None,
        segment_random=False
    ):
        """
        Given an segment ID and other spectral feature variables,
        return a spectro-temporal representation of the segment (e.g., MFCCs)
        Args:
            uttr_id (str): segment ID, i.e., the name of the wav file
            num_frames (int): length of the (MFCC) vector sequence (in frames)
            max_num_frames (int): max length of the (MFCC) vector sequence
            spectral_dim (int): the num of spectral components (default 13)
            start_index (int): the index of the 1st component (default: 0)
            end_index (int): the index of the last component (default: 13)
            segment_random (bool): whether to take a random segment from signal

        Returns:
            speech spectro-temporal representation (torch.Tensor: 300 x 13)
        """
        # these were added to enable differet uttr lengths during inference
        if num_frames is None: num_frames = self.num_frames
        if max_num_frames is None: max_num_frames = self.max_num_frames
        if spectral_dim is None: spectral_dim = self.spectral_dim
        if start_index is None: start_index = self.start_index
        if end_index is None: end_index = self.end_index

        # path to feature vector sequence (normalized)
        file_name = self.data_dir + uttr_id + '.' + \
            self.feature_type.lower() + '.norm.npy'

        # load normalized feature vector sequence from desk
        spectral_seq  = np.load(file_name)

        # sampling is used to get a random segment from the speech signal
        # by default random segmentation is disabled
        if segment_random:
            # sample N frames from the utterance
            uttr_len = spectral_seq.shape[1]   # utterance length in frames

            # if the signal is shorter than num_frames, take it as it is
            # this was added this for short utterances in DEV, EVA set
            if uttr_len - num_frames <= 0:
                sample_start = 0
                num_frames = uttr_len
            else:
                # beginning of the random speech sample
                sample_start = random.randrange(uttr_len - num_frames)

            sample_end = sample_start + num_frames # e.g. 154 + 300 (3-sec. )
            spectral_sample = spectral_seq[start_index:end_index,
                sample_start:sample_end]

        else: # if no random segmentation, i.e., during inference
            spectral_sample = spectral_seq[start_index:end_index, :num_frames]


        # convert to pytorch tensor
        spectral_tensor = torch.from_numpy(spectral_sample)

        # apply padding to the speech sample represenation
        spectral_tensor_pad = torch.zeros(spectral_dim, max_num_frames)

        # this step controls both x-axis (frames) and y-axis (spectral coefs.)
        # for example, when only 13 coefficients are used, then use
        # spectral_tensor_pad[:13,:n_frames] = spectral_tensor[:13,:n_frames]
        # likewise, the speech signal can be sampled (frame-level) as
        # spectral_tensor_pad[:feat_dim,:25] = spectral_tensor[:feat_dim,:25]

        # sample a random start index
        _start_idx = random.randrange(1 + max_num_frames - num_frames)

        # to deal with short utterances in DEV and EVA splits
        num_frames = min(spectral_seq.shape[1], num_frames)

        spectral_tensor_pad[:spectral_dim,_start_idx:_start_idx + num_frames] = \
            spectral_tensor[:spectral_dim,:num_frames]


        return spectral_tensor_pad.float() # convert to float tensor


    def transform_label_y(self, label):
        """
        Given the label of data point (language), return index (int)
        e.g.,  'RUS' --> 4
        """
        return self.label2index[label] # index of the label in the featurizer


##### CLASS SpeechDataset: A class to handle (batch) speech transformation
class SpeechDataset(Dataset):
    def __init__(self, speech_df, featurizer):
        """
        Args:
            speech_df (pandas.df): a pandas dataframe (label, split, file)
            featurizer (SpeechFeaturizer): the speech featurizer
        """
        self.speech_df = speech_df
        self._featurizer = featurizer

        # read data and make splits
        self.train_df = self.speech_df[self.speech_df.split=='TRA']
        self.train_size = len(self.train_df)

        self.val_df = self.speech_df[self.speech_df.split=='DEV']
        self.val_size = len(self.val_df)

        self.test_df = self.speech_df[self.speech_df.split=='EVA']
        self.test_size = len(self.test_df)

        # print('Size of the splits (train, val, test): ',  \
        #     self.train_size, self.val_size, self.test_size)

        self._lookup_dict = {
            'TRA': (self.train_df, self.train_size),
            'DEV': (self.val_df, self.val_size),
            'EVA': (self.test_df, self.test_size)
        }

        # by default set mode to train
        self.set_mode(split='TRA')

        # this was added to differentiate between training & inference
        self.debug_mode = None


    def set_mode(self, split='TRA'):
         """Set the mode using the split column in the dataframe. """
         self._target_split = split
         self._target_df, self._target_size = self._lookup_dict[split]


    def __len__(self):
        "Returns the number of the data points in the target split."
        return self._target_size


    def __getitem__(self, index):
        """A data transformation logic for one data point in the dataset.
        Args:
            index (int): the index to the data point in the target dataframe
        Returns:
            a dictionary holding the point representation, e.g.,
                signal (x_data), label (y_target)
        """
        uttr = self._target_df.iloc[index]

        # to enable random segmentation during training
        is_training = (self._target_split=='TRA')

        spectral_sequence = self._featurizer.transform_input_X(uttr.uttr_id,
            segment_random = is_training,
            num_frames=None, # it is important to set this to None
            spectral_dim=None
        )

        label_idx = self._featurizer.transform_label_y(uttr.language)

        return {
            'x_data': spectral_sequence,
            'y_target': label_idx,
            #'uttr_id': uttr.uttr_id
        }


    def get_num_batches(self, batch_size):
        """
        Given batch size (int), return the number of dataset batches (int)
        """
        return math.ceil((len(self) / batch_size))


##### A METHOD TO GENERATE BATCHES WITH A DATALOADER WRAPPER
def generate_batches(speech_dataset, batch_size, shuffle_batches=True,
    drop_last_batch=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader and ensures that
      each tensor is on the right device (i.e., CPU or GPU).
    """
    dataloader = DataLoader(dataset=speech_dataset, batch_size=batch_size,
        shuffle=shuffle_batches, drop_last=drop_last_batch)#drop_last_batch

    # for each batch, yield a dictionay with keys: x_data, y_target
    for data_dict in dataloader:
        # an dict object to yield in each iteration
        batch_data_dict = {}

        for var_key in data_dict:
            # when using uttr_id in data_dict, keep uttr_id on CPU and not GPU
            if var_key != 'uttr_id':
                batch_data_dict[var_key] = data_dict[var_key].to(device)
            else:
                batch_data_dict[var_key] = data_dict[var_key]

        yield batch_data_dict


##### CLASS FrameDropout: A custome layer for frame dropout
class FrameDropout(nn.Module):
    def __init__(self, dropout_prob=0.2):
        """Applies dropout on the frame level so entire feature vector will be
            evaluated to zero vector with probability p.
        Args:
            p (float): dropout probability
        """
        super(FrameDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, spectral_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_frame_idx = [i for i in range(sequence_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, :, drop_frame_idx] = 0

        return x_in


##### CLASS SpectralDropout: A custome layer for spectral (coefficient) dropout
class SpectralDropout(nn.Module):
    def __init__(self, dropout_prob=0.2, feature_idx=None):
        """Applies dropout on the feature level so spectral component accross
             vectors are replaced with zero (row-)vector with probability p.
        Args:
            p (float): dropout probability
            feature_idx (int): to mask specific spectral coeff. during inference
        """
        super(SpectralDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, spectral_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_feature_idx = [i for i in range(spectral_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, drop_feature_idx, :] = 0

        return x_in


##### CLASS FrameReverse: A custome layer for frame sequence reversal
class FrameReverse(nn.Module):
    def __init__(self):
        """Reverses the frame sequence in the input signal. """
        super(FrameReverse, self).__init__()

    def forward(self, x_in):
        batch_size, spectral_dim, sequence_dim = x_in.shape
        # reverse indicies
        reversed_idx = [i for i in reversed(range(sequence_dim))]
        x_in[:, :, reversed_idx] = x_in

        return x_in


##### CLASS FrameShuffle: A custome layer for frame sequence shuflle
class FrameShuffle(nn.Module):
    def __init__(self):
        """Shuffle the frame sequence in the input signal, given a bag size. """
        super(FrameShuffle, self).__init__()

    def forward(self, x_in, bag_size=1):
        batch_size, spectral_dim, seq_dim = x_in.shape

        # shuffle idicies according to bag of frames size
        # make the bags of frames
        seq_idx = list(range(seq_dim))

        # here, a list of bags (lists) will be made
        frame_bags = [seq_idx[i:i+bag_size] for i in range(0, seq_dim, bag_size)]

        # shuffle the bags
        random.shuffle(frame_bags)

        # flatten the bags into a sequential list
        shuffled_idx = [idx for bag in frame_bags for idx in bag]

        x_in[:, :, shuffled_idx] = x_in

        return x_in


##### CLASS ConvSpeechEncoder: A multi-layer convolutional encoder
class ConvSpeechEncoder(nn.Module):
    """A 1D 3-layer convolutional encoder for speech data."""
    def __init__(self,
        spectral_dim=13,
        max_num_frames= 384,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        signal_dropout_prob=0.2,
        dropout_frames=False,
        dropout_spectral_features=False,
        mask_signal=False
    ):
        """
        Args:
            spectral_dim (int): number of spectral coefficients
            max_num_frames (int): max number of acoustic frames in input
            num_channels (list): number of channels per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_dropout_prob (float): signal dropout probability, either
                frame dropout or spectral feature dropout
            signal_masking (bool):  whether to mask signal during inference

            How to use example:
            speech_enc = ConvSpeechEncoder(
                spectral_dim=13,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',

                # this will apply frame dropout with 0.2 prob
                signal_dropout_prob=0.2,
                dropout_frames=True,
                dropout_spectral_features=False,
                mask_signal= False
            ):
        """
        super(ConvSpeechEncoder, self).__init__()
        self.spectral_dim = spectral_dim
        self.max_num_frames = max_num_frames
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.dropout_frames = dropout_frames
        self.dropout_spectral_features = dropout_spectral_features
        self.mask_signal = mask_signal

        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enableed
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_spectral_features: # if spectral dropout is enabled
            self.signal_dropout = SpectralDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()


        # Convolutional layer  1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.spectral_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional layer  2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional layer  3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # determine the output dimensionality of the resulting tensor
            shrinking_dims = sum([(i - 1) for i in filter_sizes])
            out_dim = self.max_num_frames - shrinking_dims

            self.PoolLayer = nn.MaxPool1d(kernel_size=out_dim, stride=1) # 362
        else:
            #TODO: implement other statistical pooling approaches
            raise NotImplementedError


    def forward(self,
        x_in,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        shuffle_frames=False,
        shuffle_bag_size= 1
    ):
        """The forward pass of the speech encoder

        Args:
            x_in (torch.Tensor): an input data tensor with the shape
                (batch_size, spectral_dim, max_num_frames)
            frame_dropout (bool): whether to mask out frames (inference)
            feature_dropout (bool): whether to mask out features (inference)
            frame_reverse (bool): whether to reverse frames (inference)
            shuffle_frames (bool): whether to shuffle frames (train & inf.)
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # apply signal dropout on the input (if any)
        # signal dropout, disabled on evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)

        # signal masking during inference (explicit)
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and shuffle_frames:
            x_in = self.frame_shuffle(x_in, bag_size=shuffle_bag_size)

        # apply the convolutional transformations on the signal
        conv1_f = self.conv1(x_in)
        conv2_f = self.conv2(conv1_f)
        conv3_f = self.conv3(conv2_f)

        # max pooling
        conv_features = self.PoolLayer(conv3_f).squeeze(dim=2)

        return conv_features


##### CLASS FeedforwardClassifier: multi-layer feed-forward classifier
class FeedforwardClassifier(nn.Module):
    """A fully-connected feedforward classifier. """
    def __init__(self,
        num_classes=6,
        input_dim=512,
        hidden_dim=512,
        num_layers=3,
        unit_dropout=False,
        dropout_prob=0.0
    ):
        """
        Args:
            num_classes (int): num of classes or size the softmax layer
            input_dim (int): dimensionality of input vector
            hidden_dim (int): dimensionality of hidden layer
            node_dropout (bool): whether to apply unit dropout
            dropout_prob (float): unit dropout probability
        """
        super(FeedforwardClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.unit_dropout = unit_dropout
        self.dropout_prob = dropout_prob

        self._classifier = torch.nn.Sequential()

        # iterate over number of layers and add layer to task classifier
        for i in range(self.num_layers - 1):
            layer_dim = self.input_dim if i == 0 else self.hidden_dim

            # for the last layer, name it last_relu so it can be obtained
            if i < self.num_layers - 2:
                layer_tag = 'fc' + str(i + 1)
            else:
                layer_tag = 'fc_last'

            # add a linear transformation
            self._classifier.add_module(layer_tag,
                nn.Linear(layer_dim, self.hidden_dim))
            # add non-linearity
            self._classifier.add_module(layer_tag + "_relu", nn.ReLU())

            if self.unit_dropout:
                self._classifier.add_module(layer_tag + "_drop",
                    nn.Dropout(self.dropout_prob))

        # output layer, logits
        self._classifier.add_module("logits",
            nn.Linear(self.hidden_dim, self.num_classes))


    def forward(self,
        x_in,
        apply_softmax=False,
        return_vector=False,
        target_layer='fc_last_relu'
    ):
        """
        The forward pass of the feedforward network.

        Args:
            x_in (torch.Tensor): an input tensor, shape (batch_size, input_dim)
            apply_softmax (bool): a flag for the softmax activation, this should
                be false if used with the cross-entropy losses
        Returns:
            A tensor (torch.Tensor): logits or softmax, shape (num_classes, )
        """

        # if we need to obtain vectors (for analysis), iterate ...
        layer_vec = x_in
        if return_vector:
            for _tag, nn_layer in self._classifier._modules.items():
                layer_vec = nn_layer(layer_vec)

                if _tag == target_layer:
                    return layer_vec

        # otherwise, normal forward pass ...
        else:
            y_hat = self._classifier(x_in)

            return torch.softmax(y_hat, dim=1) if apply_softmax else y_hat


##### CLASS GradientReversal: Gradient Reversal layer for adversarial adaptation
class GradientReversal(Function):
    """GRL: forward pass --> identitiy, backward pass --> - lambda x grad """
    @staticmethod
    def forward(ctx, x_in, adap_para):
        # Store context for backprop
        ctx.adap_para = adap_para

        # Forward pass is a no-op
        return x_in.view_as(x_in)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to  - adap_para the gradient
        grad_reverse = grad_output.neg() * ctx.adap_para

        # Must return same number as inputs to forward()
        return grad_reverse, None


##### CLASS SpeechClassifier: multi-layer encoder + feed-forward network
class SpeechClassifier(nn.Module):
    """A classifier on top of speech encoder. """
    def __init__(self,
        speech_segment_encoder,
        task_classifier
    ):
        """
        Args:
            speech_segment_encoder (ConvSpeechEncoder): speech encoder model
            task_classifier (FeedforwardClassifier): n-way classifier
        """
        super(SpeechClassifier, self).__init__()
        self.speech_encoder = speech_segment_encoder
        self.task_classifier = task_classifier

    def forward(self, x_in, apply_softmax=False, return_vector=False,
        shuffle_frames=False, shuffle_bag_size=1):
        """
        The forward pass of the end-to-end classifier. Given x_in (torch.Tensor),
            return output tensor y_hat or out_vec (torch.Tensor)
        """
        conv_features = self.speech_encoder(x_in,
            shuffle_frames=shuffle_frames,
            shuffle_bag_size=shuffle_bag_size)

        if return_vector:
            out_vec =  self.task_classifier(conv_features, apply_softmax=False,
                return_vector=True, target_layer='fc_last_relu')

            return out_vec

        else:
            y_hat = self.task_classifier(conv_features, apply_softmax)

            return y_hat


##### CLASS AdaptiveSpeechClassifierI: Speech classifier with adversarial block
class AdaptiveSpeechClassifierI(nn.Module):
    """
    A classifier on top of speech encoder, with aux classifier that is
    adversarially trained for domain adaptation or (protected) attribute removal
    """
    def __init__(self,
        speech_segment_encoder,
        task_classifier,
        adversarial_classifier
    ):
        """
        Args:
            speech_segment_encoder (ConvSpeechEncoder): speech encoder model
            task_classifier (FeedforwardClassifier): n-way classifier
            adversarial_classifier (FeedforwardClassifier): aux classifier
        """
        super(AdaptiveSpeechClassifierI, self).__init__()
        self.speech_encoder = speech_segment_encoder
        self.task_classifier = task_classifier
        self._aux_classifier = adversarial_classifier

    def forward(self, x_in, apply_softmax=False, return_vector=False,
        shuffle_frames=False, grl_lambda=1.0):
        """
        The forward pass of the end-to-end classifier. Given x_in (torch.Tensor),
            return output tensor y_hat or out_vec (torch.Tensor)
        """
        conv_features = self.speech_encoder(x_in, shuffle_frames=shuffle_frames)
        _aux_features = GradientReversal.apply(conv_features, grl_lambda)

        if return_vector:
            out_vec =  self.task_classifier(conv_features, apply_softmax=False,
                return_vector=True, target_layer='fc_last_relu')

            return out_vec

        else:
            y_hat = self.task_classifier(conv_features, apply_softmax)
            d_hat = self._aux_classifier(_aux_features)

            return y_hat, d_hat


##### CLASS AdaptiveSpeechClassifierII: Speech classifier with adversarial block
class AdaptiveSpeechClassifierII(nn.Module):
    """
    A classifier on top of speech encoder, with aux classifier that is
    adversarially trained for domain adaptation or (protected) attribute removal
    """
    def __init__(self,
        speech_segment_encoder,
        task_classifier,
        adversarial_classifier,
        fc_input_dim=512,
        fc_output_dim=512
    ):
        """
        Args:
            speech_segment_encoder (ConvSpeechEncoder): speech encoder model
            task_classifier (FeedforwardClassifier): n-way classifier
            adversarial_classifier (FeedforwardClassifier): aux classifier
        """
        super(AdaptiveSpeechClassifierII, self).__init__()
        self.speech_encoder = speech_segment_encoder

        self.fc_layer = nn.Sequential(
            nn.Linear(fc_input_dim, fc_output_dim),
            nn.ReLU()
        )

        self.task_classifier = task_classifier
        self._aux_classifier = adversarial_classifier


    def forward(self, x_in, apply_softmax=False, return_vector=False,
        shuffle_frames=False, grl_lambda=1.0):
        """
        The forward pass of the end-to-end classifier. Given x_in (torch.Tensor),
            return output tensor y_hat or out_vec (torch.Tensor)
        """
        conv_features = self.speech_encoder(x_in, shuffle_frames=shuffle_frames)
        fc_features   = self.fc_layer(conv_features)
        _aux_features = GradientReversal.apply(fc_features, grl_lambda)

        if return_vector:
            out_vec =  self.task_classifier(fc_features, apply_softmax=False,
                return_vector=True, target_layer='fc_last_relu')

            return out_vec

        else:
            y_hat = self.task_classifier(fc_features, apply_softmax)
            d_hat = self._aux_classifier(_aux_features)

            return y_hat, d_hat
