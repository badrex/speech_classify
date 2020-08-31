# coding: utf-8

import numpy as np
import random

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Function


##### CLASS SpeechFeaturizer
class SpeechFeaturizer(object):
    """ The Featurizer handles the low-level speech features and labels. """

    def __init__(self,
        data_dir,
        feature_type,
        label_set,
        max_num_frames,
        num_frames,
        feature_dim=13,
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
            feature_dim (int): the num of spectral components (default 13)
            start_index (int): the index of the 1st component (default: 0)
            end_index (int): the index of the last component (default: 13)
        """
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        self.start_index = start_index
        self.end_index = end_index

        # get set of the labels in the dataset
        self.label_set = label_set

        # obtain index --> label dict
        self.index2label = {idx:lbl for (lbl, idx) in enumerate(self.label_set)}


        # obtain label --> index dict
        self.label2index = {idx: lbl for (idx, lbl) in self.index2label.items()}



    def transform_input_X(self,
        uttr_id,
        num_frames=None,
        max_num_frames=None,
        feature_dim=None,
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
            feature_dim (int): the num of spectral components (default 13)
            start_index (int): the index of the 1st component (default: 0)
            end_index (int): the index of the last component (default: 13)
            segment_random (bool): whether to take a random segment from signal

        Returns:
            speech spectro-temporal representation (torch.Tensor: 300 x 13)
        """
        # these were added to enable differet uttr lengths during inference
        if num_frames is None: num_frames = self.num_frames
        if max_num_frames is None: max_num_frames = self.max_num_frames
        if feature_dim is None: feature_dim = self.feature_dim
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
        spectral_tensor_pad = torch.zeros(feature_dim, max_num_frames)

        # this step controls both x-axis (frames) and y-axis (spectral coefs.)
        # for example, when only 13 coefficients are used, then use
        # spectral_tensor_pad[:13,:n_frames] = spectral_tensor[:13,:n_frames]
        # likewise, the speech signal can be sampled (frame-level) as
        # spectral_tensor_pad[:feat_dim,:25] = spectral_tensor[:feat_dim,:25]

        # sample a random start index
        _start_idx = random.randrange(1 + max_num_frames - num_frames)

        # to deal with short utterances in DEV and EVA splits
        num_frames = min(spectral_seq.shape[1], num_frames)

        spectral_tensor_pad[:feature_dim,_start_idx:_start_idx + num_frames] = \
            spectral_tensor[:feature_dim,:num_frames]


        return spectral_tensor_pad.float() # convert to float tensor


    def transform_label_y(self, label):
        """
        Given the label of the data point (language), return label index
        Args:
            label (str): the target label in the dataset (e.g., 'RUS')
        Returns:
            the index of the label in the featurizer
        """
        return self.label2index[label]


##### CLASS LID_Dataset
class LID_Dataset(Dataset):
    def __init__(self, speech_df, vectorizer):
        """
        Args:
            speech_df (pandas.df): a pandas dataframe (label, split, file)
            vectorizer (SpeechFeaturizer): the speech vectorizer
        """
        self.speech_df = speech_df
        self._vectorizer = vectorizer

        # read data and make splits
        self.train_df = self.speech_df[self.speech_df.split=='TRA']
        self.train_size = len(self.train_df)

        self.val_df = self.speech_df[self.speech_df.split=='DEV']
        self.val_size = len(self.val_df)

        self.test_df = self.speech_df[self.speech_df.split=='EVA']
        self.test_size = len(self.test_df)

        print(self.train_size, self.val_size, self.test_size)

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
        return self._target_size


    def __getitem__(self, index):
        """Data transformation logic for one data point.
        Args:
            index (int): the index to the data point in the dataframe
        Returns:
            a dictionary holding the point representation:
                signal (x_data), label (y_target), and uttr ID (uttr_id)
        """
        uttr = self._target_df.iloc[index]

        # enable random segmentation during training
        is_training = (self._target_split=='TRA')

        feature_sequence = self._vectorizer.transform_input_X(uttr.uttr_id,
            segment_random = is_training,
            num_frames=None, # it is important to set this to None
            feature_dim=None
        )

        lang_idx = self._vectorizer.transform_label_y(uttr.language)

        return {
            'x_data': feature_sequence,
            'y_target': lang_idx,
            'uttr_id': uttr.uttr_id
        }


    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


##### A METHOD TO GENERATE BATCHES
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            if name != 'uttr_id':
                out_data_dict[name] = data_dict[name].to(device)
            else:
                out_data_dict[name] = data_dict[name]
        yield out_data_dict


##### A custome layer for frame dropout
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
        batch_size, feature_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_frame_idx = [i for i in range(sequence_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, :, drop_frame_idx] = 0

        return x_in


##### A custome layer for frame dropout
class FeatureDropout(nn.Module):
    def __init__(self, dropout_prob=0.2, feature_idx=None):
        """Applies dropout on the feature level so feature accross vectors are
            are replaced with zero vector with probability p.
        Args:
            p (float): dropout probability
        """
        super(FeatureDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, x_in):
        batch_size, feature_dim, sequence_dim = x_in.shape

        # randomly sample frame indecies to be dropped
        drop_feature_idx = [i for i in range(feature_dim) \
            if torch.rand(1).item() < self.dropout_prob]

        x_in[:, drop_feature_idx, :] = 0

        return x_in


##### A custome layer for frame sequence reversal
class FrameReverse(nn.Module):
    def __init__(self):
        """Reverses the frame sequence in the input signal. """
        super(FrameReverse, self).__init__()

    def forward(self, x_in):
        batch_size, feature_dim, sequence_dim = x_in.shape
        # reverse indicies
        reversed_idx = [i for i in reversed(range(sequence_dim))]
        x_in[:, :, reversed_idx] = x_in

        return x_in


##### A custome layer for frame sequence shuflle
class FrameShuffle(nn.Module):
    def __init__(self):
        """Shuffle the frame sequence in the input signal, given a bag size. """
        super(FrameShuffle, self).__init__()

    def forward(self, x_in, bag_size):
        batch_size, feature_dim, seq_dim = x_in.shape

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


##### A Convolutional model: Spoken Language Identifier
class ConvNet_LID(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()



        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=384-22, stride=1) # 362
        else:
            raise NotImplementedError

        # Fully conntected layers block
        self.fc_layers = torch.nn.Sequential()

        if bottleneck:
            # Bottleneck layer
            self.fc_layers.add_module("fc_bn",
                nn.Linear(num_channels[2], self.bottleneck_size))
            self.fc_layers.add_module("relu_bn", nn.ReLU())

            # then project to higher dim
            self.fc_layers.add_module("fc2",
                nn.Linear(self.bottleneck_size, self.output_dim))
            self.fc_layers.add_module("relu_fc2", nn.ReLU())

        else:
            # then project to two identical fc layers
            #self.fc_layers.add_module("fc1", nn.Linear(512, 512))
            #self.fc_layers.add_module("relu_fc1", nn.ReLU())

            self.fc_layers.add_module("fc2",
                nn.Linear(num_channels[2], self.output_dim))
            self.fc_layers.add_module("relu_fc2", nn.ReLU())

        # Output fully connected --> softmax
        self.fc_layers.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        z1 = self.ConvLayer1(x_in)
        z2 = self.ConvLayer2(z1)
        z3 = self.ConvLayer3(z2)

        # max pooling
        #print(z3.shape)
        z4 = self.PoolLayer(z3).squeeze(dim=2)

        # if we need to analyze bottle neck feature, go into this code block
        if return_bn:
            feature_vector = z4
            for _name, module in self.fc_layers._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_bn':
                    return feature_vector

        else:
            y_out = self.fc_layers(z4)

            # softmax
            if apply_softmax:
                y_out = torch.softmax(y_out, dim=1)

            return y_out


##### An LSTM-based recurrent model: Spoken Language Identifier
class LSTMNet_LID(nn.Module):
    def __init__(self,
        feature_dim=40,
        output_dim=64,
        hidden_dim=128,
        num_classes=6,
        bottleneck_size=64,
        n_layers=2,
        unit_dropout_prob=0.0,
        signal_dropout_prob=0.0
    ):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer


            Usage example:
            model = LSTM_LID(feature_dim=40,
                num_classes=6
            )
        """
        super(LSTMNet_LID, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, n_layers, dropout=unit_dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(unit_dropout_prob)

        # fully connected block
        self.fc_layers = torch.nn.Sequential()

        # 512 x 64
        self.fc_layers.add_module("linear_fc_bn", nn.Linear(hidden_dim, bottleneck_size))
        self.fc_layers.add_module("relu_fc_bn", nn.ReLU())

        # 64 x 512
        self.fc_layers.add_module("linear_fc_2", nn.Linear(bottleneck_size, output_dim))
        self.fc_layers.add_module("relu_fc_2", nn.ReLU())

        # 64 x 6
        self.fc_layers.add_module("y_out", nn.Linear(output_dim, num_classes))


    def forward(self, x_in, hidden_0, apply_softmax=False, return_vector=False, frame_dropout=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # permute input for recurrent computation
        x_in = x_in.permute(0, 2, 1)

        #x_in = x_in.long()
        #embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(x_in, hidden_0)
        #print('lstm_out 1', lstm_out.shape)

        last_hidden = lstm_out[:, -1, :]
        #print('last_hidden', last_hidden.shape)

        #out = self.dropout(lstm_out)
        #z = self.fc_layers(lstm_out)
        #print('z', z.shape)

        if return_vector:
            feature_vector = last_hidden

            for _name, module in self.fc_layers._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_fc_2':
                    return feature_vector

        else:

            y_out = self.fc_layers(last_hidden)
            #print(y_out.shape)

        # softmax
        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        return y_out, hidden

    def init_hidden(self, batch_size, device):
        """
        Given batch size & the device where the training of NN is taking place,
        return a proper (zero) initialiization for the LSTM model as (h_0, c_0).
        """
        hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        )
        return hidden


##### Diagnostic classifiers
class LinearLID(torch.nn.Module):
    def __init__(self,
        feature_dim=40,
        num_frames=300,
        num_classes=6
    ):
        super(LinearLID, self).__init__()
        self.linear = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):

        # avergare the frames
        x_in = torch.mean(x_in, dim=2)
        #print(x_in.shape)
        y_out = self.linear(x_in)

        # softmax
        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        return y_out

class MLPNetLID(torch.nn.Module):
    def __init__(self,
        feature_dim=13,
        num_frames=300,
        hidden_dim=512,
        num_layers=2,
        num_classes=6
    ):

        super(MLPNetLID, self).__init__()
        self.fc_layers = torch.nn.Sequential()

        for i in range(num_layers):
            layer_id = 'fc_' + str(i + 1)
            if i == 0:
                self.fc_layers.add_module(layer_id + '_linear', nn.Linear(feature_dim, hidden_dim))
            else:
                self.fc_layers.add_module(layer_id + '_linear', nn.Linear(hidden_dim, hidden_dim))

            self.fc_layers.add_module(layer_id + '_relu', nn.ReLU())

        self.fc_layers.add_module("y_out", nn.Linear(hidden_dim, num_classes))


    def forward(self, x_in, apply_softmax=False):

        # avergare the frames
        x_in = torch.mean(x_in, dim=2)

        #print(x_in.shape)

        y_out = self.fc_layers(x_in)

        # softmax
        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        return y_out


class MLPNetLID2(torch.nn.Module):
    def __init__(self,
        feature_dim=13,
        num_frames=300,
        hidden_dim=512,
        num_layers=2,
        num_classes=6
    ):

        super(MLPNetLID2, self).__init__()
        self.fc_layers = torch.nn.Sequential()

        for i in range(num_layers):
            layer_id = 'fc_' + str(i + 1)
            if i == 0:
                self.fc_layers.add_module(layer_id + '_linear', nn.Linear(num_frames, hidden_dim))
            else:
                self.fc_layers.add_module(layer_id + '_linear', nn.Linear(hidden_dim, hidden_dim))

            self.fc_layers.add_module(layer_id + '_relu', nn.ReLU())

        self.fc_layers.add_module("y_out", nn.Linear(hidden_dim, num_classes))


    def forward(self, x_in, apply_softmax=False):

        # avergare the frames
        x_in = x_in[:, [0], :].squeeze(dim=1)#torch.mean(x_in, dim=2)

        #print(x_in.shape)


        y_out = self.fc_layers(x_in)

        #print(y_out.shape)

        # softmax
        if apply_softmax:
            y_out = torch.softmax(y_out, dim=1)

        return y_out


# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha

        # Must return same number as inputs to forward()
        return output, None


##### A Convolutional model: Spoken Language Identifier
class ConvNet_LID_DA(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.language_classifier = torch.nn.Sequential()

        self.language_classifier.add_module("fc_bn",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.language_classifier.add_module("relu_bn", nn.ReLU())

        # then project to higher dim
        self.language_classifier.add_module("fc2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.language_classifier.add_module("relu_fc2", nn.ReLU())

        # Output fully connected --> softmax
        self.language_classifier.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_channels[2], 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(1024, 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

        #self.signal_reverse_classifier = nn.Sequential(
        #    nn.Linear(num_channels[2], 256), #nn.BatchNorm1d(100),
        #    nn.ReLU(),
        #    nn.Linear(256, 2)
        #)


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in) # if self.eval
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        z1 = self.ConvLayer1(x_in)
        z2 = self.ConvLayer2(z1)
        z3 = self.ConvLayer3(z2)

        # max pooling
        features = self.PoolLayer(z3).squeeze(dim=2)

        # if we need to analyze bottle neck feature, go into this code block
        # if return_bn:
        #     feature_vector = features
        #     for _name, module in self.language_classifier._modules.items():
        #         feature_vector = module(feature_vector)
        #
        #         if _name == 'relu_bn':
        #             return feature_vector

        # else:

        reverse_features = GradientReversalFn.apply(features, grl_lambda)

        class_pred = self.language_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        #sigrev_pred = self.signal_reverse_classifier(features

        # softmax
        if apply_softmax:
            class_pred = torch.softmax(class_pred, dim=1)

        return class_pred, domain_pred # or torch.sigmoid()


##### A Convolutional model: Spoken Language Identifier, with unit dropout
class ConvNet_LID_DA_wDropout(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        unit_dropout_prob=0.5,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA_wDropout, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.unit_dropout_prob = unit_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.language_classifier = torch.nn.Sequential()

        self.language_classifier.add_module("fc_bn",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.language_classifier.add_module("relu_bn", nn.ReLU())
        self.language_classifier.add_module("drop_bn", nn.Dropout(self.unit_dropout_prob))


        # then project to higher dim
        self.language_classifier.add_module("fc2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.language_classifier.add_module("relu_fc2", nn.ReLU())
        self.language_classifier.add_module("drop_fc2", nn.Dropout(self.unit_dropout_prob))


        # Output fully connected --> softmax
        self.language_classifier.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_channels[2], 256), #nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(self.unit_dropout_prob),
            nn.Linear(256, 2)
        )


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        z1 = self.ConvLayer1(x_in)
        z2 = self.ConvLayer2(z1)
        z3 = self.ConvLayer3(z2)

        # max pooling
        features = self.PoolLayer(z3).squeeze(dim=2)

        # if we need to analyze bottle neck feature, go into this code block
        # if return_bn:
        #     feature_vector = features
        #     for _name, module in self.language_classifier._modules.items():
        #         feature_vector = module(feature_vector)
        #
        #         if _name == 'relu_bn':
        #             return feature_vector

        # else:

        reverse_features = GradientReversalFn.apply(features, grl_lambda)

        class_pred = self.language_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)

        # softmax
        if apply_softmax:
            class_pred = torch.softmax(class_pred, dim=1)

        return class_pred, domain_pred # or torch.sigmoid()



##### A Convolutional model: Spoken Language Identifier, with unit dropout
class ConvNet_LID_DA_2(nn.Module):
    def __init__(self,
        feature_dim=14,
        num_classes=6,
        bottleneck=False,
        bottleneck_size=64,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        #unit_dropout_prob=0.5,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA_2, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        #self.unit_dropout_prob = unti_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        if self.dropout_frames: # if frame dropout is enables
            self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        elif self.dropout_features: # if frame dropout is enables
            self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        self.frame_reverse = FrameReverse()

        # frame reversal layer
        self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.fc_layers = torch.nn.Sequential()

        self.fc_layers.add_module("fc1",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.fc_layers.add_module("relu_fc1", nn.ReLU())

        #self.fc_classifier.add_module("fc2",
        #    nn.Linear(self.bottleneck_size, self.output_dim))
        #self.fc_classifier.add_module("relu_fc2", nn.ReLU())
        #self.fc_classifier.add_module("drop_bn", nn.Dropout(self.unit_dropout_prob))


        self.label_classifier = torch.nn.Sequential()

        self.label_classifier.add_module("fc2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.label_classifier.add_module("relu_fc2", nn.ReLU())
        #self.label_classifier.add_module("drop_fc2", nn.Dropout(self.unit_dropout_prob))

        # Output fully connected --> softmax
        self.label_classifier.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_channels[2], 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            #nn.Dropout(self.unit_dropout_prob),
            nn.Linear(1024, 1024), #nn.BatchNorm1d(100),
            nn.ReLU(),
            #nn.Dropout(self.unit_dropout_prob),
            nn.Linear(1024, 2)
        )


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        if self.training:
            x_in = self.signal_dropout(x_in)


        # signal masking during inference
        if self.eval and self.mask_signal:
            x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        if self.eval and frame_reverse: x_in = self.frame_reverse(x_in)
        if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        z1 = self.ConvLayer1(x_in)
        z2 = self.ConvLayer2(z1)
        z3 = self.ConvLayer3(z2)

        # max pooling
        conv_features = self.PoolLayer(z3).squeeze(dim=2)

        # fc features
        fc_features = self.fc_layers(conv_features)

        #
        # if we need to analyze bottle neck feature, go into this code block
        if return_bn:
            feature_vector = fc_features

            for _name, module in self.label_classifier._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_fc2':
                    return feature_vector



        reverse_fc_features = GradientReversalFn.apply(fc_features, grl_lambda)

        class_pred = self.label_classifier(fc_features)
        domain_pred = self.domain_classifier(reverse_fc_features)

        # softmax
        if apply_softmax:
            class_pred = torch.softmax(class_pred, dim=1)

        return class_pred, domain_pred



#### New Adv ConvNet with stronger domain classifier

##### A Convolutional model: Spoken Language Identifier
class ConvNet_LID_DA_3(nn.Module):
    def __init__(self,
        feature_dim=13,
        num_classes=6,
        bottleneck=True,
        bottleneck_size=512,
        signal_dropout_prob=0.2,
        num_channels=[128, 256, 512],
        filter_sizes=[5, 10, 10],
        stride_steps=[1, 1, 1],
        output_dim=512,
        pooling_type='max',
        dropout_frames=False,
        dropout_features=False,
        mask_signal=False):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer
            bottleneck (bool): whether or not to have bottleneck layer
            bottleneck_size (int): the dim of the bottleneck layer
            signal_dropout_prob (float): signal dropout probability, either
                frame drop or spectral feature drop
            num_channels (list): number of channeös per each Conv layer
            filter_sizes (list): size of filter/kernel per each Conv layer
            stride_steps (list): strides per each Conv layer
            pooling (str): pooling procedure, either 'max' or 'mean'
            signal_masking (bool):  whether or no to mask signal during inference

            Usage example:
            model = ConvNet_LID(feature_dim=13,
                num_classes=6,
                bottleneck=False,
                bottleneck_size=64,
                signal_dropout_prob=0.2,
                num_channels=[128, 256, 512],
                filter_sizes=[5, 10, 10],
                stride_steps=[1, 1, 1],
                pooling_type='max',
                mask_signal: False
            )
        """
        super(ConvNet_LID_DA_3, self).__init__()
        self.feature_dim = feature_dim
        self.signal_dropout_prob = signal_dropout_prob
        self.pooling_type = pooling_type
        self.bottleneck_size = bottleneck_size
        self.output_dim = output_dim
        self.dropout_frames = dropout_frames
        self.dropout_features = dropout_features
        self.mask_signal = mask_signal


        # signal dropout_layer
        #if self.dropout_frames: # if frame dropout is enables
            #self.signal_dropout = FrameDropout(self.signal_dropout_prob)

        #elif self.dropout_features: # if frame dropout is enables
            #self.signal_dropout = FeatureDropout(self.signal_dropout_prob)

        # frame reversal layer
        #self.frame_reverse = FrameReverse()

        # frame reversal layer
        #self.frame_shuffle = FrameShuffle()

        # Convolutional Block 1
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                out_channels=num_channels[0],
                kernel_size=filter_sizes[0],
                stride=stride_steps[0]),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )

        # Convolutional Block 2
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=filter_sizes[1],
                stride=stride_steps[1]),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU()
        )

        # Convolutional Block 3
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=filter_sizes[2],
                stride=stride_steps[2]),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU()
        )

        if self.pooling_type == 'max':
            # NOTE: the MaxPool kernel size 362 was determined
            # after examining the dataflow in the network and
            # observing the resulting tensor shapes
            self.PoolLayer = nn.MaxPool1d(kernel_size=362, stride=1)
        else:
            raise NotImplementedError

        # Fully conntected layers block - Language classifier
        self.task_classifier = torch.nn.Sequential()

        self.task_classifier.add_module("fc_1",
            nn.Linear(num_channels[2], self.bottleneck_size))
        self.task_classifier.add_module("relu_1", nn.ReLU())

        # then project to higher dim
        self.task_classifier.add_module("fc_2",
            nn.Linear(self.bottleneck_size, self.output_dim))
        self.task_classifier.add_module("relu_2", nn.ReLU())

        # Output fully connected --> softmax
        self.task_classifier.add_module("y_out",
            nn.Linear(self.output_dim, num_classes))

        self.domain_classifier = nn.Sequential(
            nn.Linear(2*num_channels[2], 512), #nn.BatchNorm1d(100),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, 2)
        )


    def forward(self,
        x_in,
        apply_softmax=False,
        return_bn=False,
        frame_dropout=False,
        feature_dropout=False,
        frame_reverse=False,
        frame_shuffle=False,
        shuffle_bag_size= 1,
        grl_lambda=1.0
    ):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # the feature representation x_in has to go through the following
        # transformations: 3 Convo layers, 1 MaxPool layer, 3 FC, then softmax

        # signal dropout, disabled when evaluating unless explicitly asked for
        #if self.training:
            #x_in = self.signal_dropout(x_in)


        # signal masking during inference
        #if self.eval and self.mask_signal:
        #    x_in = self.signal_dropout(x_in)

        # signal distortion during inference
        #if self.eval and frame_reverse: x_in = self.frame_reverse(x_in) # if self.eval
        #if self.eval and frame_shuffle: x_in = self.frame_shuffle(x_in, shuffle_bag_size)

        # Convo block
        z1 = self.ConvLayer1(x_in)
        z2 = self.ConvLayer2(z1)
        z3 = self.ConvLayer3(z2)

        # max pooling
        f1 = self.PoolLayer(z3).squeeze(dim=2)

        f_vect = f1
        for _name, module in self.task_classifier._modules.items():
            f_vect = module(f_vect)

            if _name == 'relu_2':
                f3 = f_vect

            elif _name == 'relu_1':
                f2 = f_vect

            elif _name == 'y_out':
                class_pred = f_vect

        # else:

        f_reverse = GradientReversalFn.apply(torch.cat((f1, f3), dim=1), grl_lambda)

        #class_pred = y_out #self.task_classifier(f1)
        domain_pred = self.domain_classifier(f_reverse)

        # softmax
        if apply_softmax: class_pred = torch.softmax(class_pred, dim=1)

        return class_pred, domain_pred #or torch.sigmoid()


##### An LSTM-based recurrent model: Spoken Language Identifier
class BiLSTM_LID_DA(nn.Module):
    def __init__(self,
        feature_dim=13,
        output_dim=512,
        hidden_dim=256,
        num_classes=6,
        bottleneck_size=512,
        n_layers=1,
        bidirectional=True,
        unit_dropout_prob=0.0,
        signal_dropout_prob=0.0
    ):
        """
        Args:
            feature_dim (int): size of the feature vector
            num_classes (int): num of classes or size the softmax layer


            Usage example:
            model = LSTM_LID(feature_dim=40,
                num_classes=6
            )
        """
        super(BiLSTM_LID_DA, self).__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(feature_dim, hidden_dim, n_layers, dropout=unit_dropout_prob, bidirectional=True)#, batch_first=True
        #self.dropout = nn.Dropout(unit_dropout_prob)

        # task classifier
        self.task_classifier = torch.nn.Sequential()
        self.task_classifier.add_module("linear_fc1", nn.Linear(2*hidden_dim, bottleneck_size))
        self.task_classifier.add_module("relu_fc1", nn.ReLU())
        self.task_classifier.add_module("linear_fc2", nn.Linear(bottleneck_size, output_dim))
        self.task_classifier.add_module("relu_fc2", nn.ReLU())
        self.task_classifier.add_module("y_out", nn.Linear(output_dim, num_classes))


        self.domain_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 256), #nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(256, 2)
        )



    def forward(self,
        x_in,
        apply_softmax=False,
        return_vector=False,
        frame_dropout=False,
        grl_lambda=1.0):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch_size, feature_dim, dataset._max_frames)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, )
        """

        # get batch size from input
        batch_size = x_in.shape[0]

        # permute input for recurrent computation
        x_in = x_in.permute(2, 0, 1)


        output, (h_n, c_n) = self.lstm(x_in)
        #print('lstm_out 1', output.shape)

        #print('h_n', h_n.shape)

        # Seperate directions
        #output = output.view(300, batch_size, 2, self.hidden_dim) #seq_len, batch, num_directions, hidden_size
        h_n = h_n.view(self.n_layers, 2, batch_size, self.hidden_dim) # num_layers, num_directions, batch, hidden_size

        # Compare directions
        #output[-1, :, 0] == h_n[:, 0] # forward
        #output[0, :, 1] == h_n[:, 1] # backward

        h_n_forward  = h_n[:, 0][-1].squeeze()
        h_n_backward = h_n[:, 1][-1].squeeze()

        #print('h_n_forward', h_n_forward.shape)

        features = torch.cat((h_n_forward, h_n_backward), dim=1)
        #print('featues ', features.shape)

        reverse_features = GradientReversalFn.apply(features, grl_lambda)


        if return_vector:
            feature_vector = features

            for _name, module in self.task_classifier._modules.items():
                feature_vector = module(feature_vector)

                if _name == 'relu_fc_2':
                    return feature_vector

        else:
            class_pred  = self.task_classifier(features)
            domain_pred = self.domain_classifier(reverse_features)


        # softmax
        if apply_softmax:
            class_pred = torch.softmax(y_out, dim=1)

        return class_pred, domain_pred
