#! /usr/bin/env python3
# coding: utf-8

import os
import sys
import collections
import yaml

# helper function
def read_log_file(file_name):
    """From a log file, return dict with val acc scores"""

    with open(file_name) as f:
        log_file_text = f.readlines()

    model_strings = []

    for i, line in enumerate(log_file_text):
        # get all model names
        #print(line[-4:-1])

        if line[:4] == 'Best' and line[-4:-1] == 'tgt':

            line_tokens = line.split()

            best_model_str = line_tokens[1] + '_' + line_tokens[-3] + '.pth'

            model_strings.append(best_model_str)

    return model_strings

# Training Routine

# obtain yml config file from cmd line and print out content
if len(sys.argv) != 2:
	sys.exit("\nUsage: " + sys.argv[0] + " <config YAML file>\n")

config_file_path = sys.argv[1] # e.g., '/speech_cls/config_1.yml'
config_args = yaml.safe_load(open(config_file_path))
#print('YML configuration file content:')
#pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(config_args)

models_to_eval = read_log_file(config_args['log_file'])
#print(models_to_eval)

for i, _model in  enumerate(models_to_eval):

    print(f"{i+1:>2} \t {_model}")
